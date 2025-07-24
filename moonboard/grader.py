#!/usr/bin/python
#-*- coding: utf-8 -*-

import numpy as np
from torch import nn
import torch
import os
import re
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import sys
import signal
import json
import csv
try:
    import commons
    import sklearn.metrics as skmetrics
    import seaborn as sns
    import matplotlib.pyplot as plt
except:
    from . import commons
from collections import Counter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##########################################
# Constants and Mappings
##########################################
# check https://towardsdatascience.com/how-to-perform-ordinal-regression-classification-in-pytorch-361a2a095a99/
# for why this encoding is used (CORAL method)
# The grades are encoded as a binary vector where each position corresponds to a threshold passed.
GRADE_DICT = {
    "6A+":[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "6B": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "6B+":[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "6C": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "6C+":[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    "7A": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    "7A+":[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    "7B": [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    "7B+":[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    "7C": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    "7C+":[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    "8A": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    "8A+":[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    # "8B": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    # "8B+":[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
}

GRADE_DICT_REVERSE = {str(v[1:]): k for k, v in GRADE_DICT.items()}

HAND_DICT = {
    "L": 1,
    "R": 2,
}

HOLD_TYPE_DICT = {
    "jug": 1,
    "pinch": 2,
    "microjug": 3,
    "micropinch": 4,
    "crimp": 5,
    "sloper": 6,
    "pocket": 7,
}

HOLD_TEXTURE_DICT = {
    "woodc": 1,
    "white": 2,
    "black": 3,
    "woodb": 4,
    "wooda": 5,
    "yellow": 6,
}

CAN_MATCH_DICT = {
    True: 1,
    False: 2,
}

ORIENTATION_DICT = {
    "N": 1,
    "NW": 2,
    "NE": 3,
    "W": 4,
    "E": 5,
    "SW": 6,
    "SE": 7,
    "S": 8,
}

MOONBOARD_VERSION_DICT = {
    "2016": 1,
    "2017": 2,
    "2019": 3,
}

###########
# build Dataset
#############


def has_too_much_holds_together(holds):
    """Check if there are too many holds together in a boulder."""
    for hold in holds:
        neighboors = 0
        for other in holds:
            if other != hold and commons.get_distance(hold, other) < commons.INSERT_DISTANCE + 10:  # 30 cm is the threshold for "together"
                neighboors += 1
        if neighboors > 1:
            return True
    return False


def filter_dataset(dataset):
    """Filtre le dataset pour ne garder que les boulders avec des prises valides."""
    list_benchmark_setters = set()
    for boulder in dataset:
        if boulder['isBenchmark'] and boulder['setter'] not in list_benchmark_setters:
            list_benchmark_setters.add(boulder['setter'])
    filtered_dataset = list(filter(lambda boulder: (
        boulder['method'] == "Feet follow hands"
        and len(boulder['holds']) >= 2 # at least 2 holds
        and boulder["grade"] not in ["8B", "8B+"] # ignore 8B+ grades
        and boulder["userGrade"] not in ["8B", "8B+"] # ignore 8B+ grades
        and (
            boulder['setter'] in list_benchmark_setters  # benchmark setters
            or (
                boulder["userGrade"] is not None  # user grade is set
                and (not has_too_much_holds_together(boulder['holds']) or (boulder["userGrade"] or boulder["grade"]) > "6C+")
                and boulder["rating"] > (3 if (boulder["userGrade"] or boulder["grade"]) < "6C+" else 1)
                and (boulder["repeats"] > 20 or (boulder["userGrade"] or boulder["grade"]) > "6C+")
            )
        )
    ), dataset))
    grade_counts = Counter(boulder["userGrade"] or boulder["grade"] for boulder in filtered_dataset)
    print("Number of boulders per grade (before oversampling):")
    for grade in sorted(grade_counts.keys()):
        print(f"{grade}: {grade_counts[grade]}")
    mean_nb_boulder_per_grade = np.mean(list(grade_counts.values()))
    print(f"Mean number of boulders per grade: {mean_nb_boulder_per_grade:.2f}")
    oversampled_dataset = oversample_dataset_by_grade(filtered_dataset)
    grade_counts_after = Counter(b["userGrade"] or b["grade"] for b in oversampled_dataset)
    print("Number of boulders per grade (after oversampling):")
    for grade in sorted(grade_counts_after.keys()):
        print(f"{grade}: {grade_counts_after[grade]}")
    return oversampled_dataset


# New function: oversample the dataset by duplicating samples from minority classes
def oversample_dataset_by_grade(dataset):
    from collections import defaultdict
    import random
    grade_to_boulders = defaultdict(list)
    for b in dataset:
        grade = b["userGrade"] or b["grade"]
        grade_to_boulders[grade].append(b)
    max_count = max(len(lst) for lst in grade_to_boulders.values())
    oversampled = []
    for grade, boulders in grade_to_boulders.items():
        if len(boulders) < max_count:
            # Sample with replacement to reach max_count
            oversampled.extend(boulders)
            oversampled.extend(random.choices(boulders, k=max_count - len(boulders)))
        else:
            oversampled.extend(boulders)
    random.shuffle(oversampled)
    return oversampled


class BoulderDataset(Dataset):
    def __init__(self, dataset, holds_data=None, input_size=11):
        self.input_size = input_size  # Number of features per hold (increased from 10 to 11)
        self.data, self.labels = [], []
        for boulder in dataset:
            boulder_vector = self.vectorize_boulder(boulder, self.input_size, holds_data)
            label_vector = GRADE_DICT[boulder["userGrade"] or boulder['grade']][1:]  # remove the first element (0) to match the output size
            self.data.append(torch.tensor(boulder_vector, dtype=torch.float32).to(device))
            self.labels.append(torch.tensor(label_vector, dtype=torch.float32).to(device))

    @staticmethod
    def vectorize_boulder(boulder, input_size, holds_data={}):
        """Convertit une beta en vecteur de difficulté des prises utilisées."""
        vector = np.zeros((14, input_size))  # Maximum number of moves in a beta (app limit 14 holds )

        for idx, hold_name in enumerate(boulder["holds"]):
            hold_data = holds_data[boulder['version']][hold_name]
            hold_type = hold_data['type']
            hold_texture = hold_data['texture']
            can_match = hold_data['can_match']
            orientation = hold_data['orientation']
            is_start = hold_name in boulder["start"]
            is_end = hold_name in boulder["end"]
            hold_pos = commons.hold_name_to_pos(hold_name)

            # Calculate distance to previous hold in "hold units"
            # for the first hold, we set distance to 0
            distance = commons.get_distance(hold_name, boulder["holds"][idx - 1]) / commons.INSERT_DISTANCE if idx > 0 else 0

            vector[idx] = [
                int(hold_pos[0]),
                int(hold_pos[1]),
                int(is_start),
                int(is_end),
                int(HOLD_TYPE_DICT[hold_type]),
                int(HOLD_TEXTURE_DICT[hold_texture]),
                int(ORIENTATION_DICT[orientation]),
                int(CAN_MATCH_DICT[can_match]),
                int(round(commons.get_hold_difficulty("L", hold_data))),
                int(round(commons.get_hold_difficulty("R", hold_data))),
                int(round(distance)),
            ]
            if input_size == 12:
                vector.append(MOONBOARD_VERSION_DICT[boulder["version"]])
        return vector

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


#############
# Classifier Model
#############
class BoulderClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(BoulderClassifier, self).__init__()
        # input_size: number of features per hold (8)
        self.input_size = input_size
        # hidden_size: number of LSTM units (128)
        self.hidden_size = hidden_size
        # num_classes: number of classes (15 for grades 6A+ to 8B+)
        self.num_classes = num_classes

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        # Fully connected layer to map LSTM output to class probabilities
        # num_classes - 1 because we are predicting the number of thresholds passed, not the class itself
        # e.g. for 15 classes, we have 14 thresholds to predict
        self.fc = nn.Linear(hidden_size, num_classes - 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take the last time step of the lstm
        return out


class FocalLoss(nn.Module):
    """
    Focal Loss for binary/multi-label classification with logits.
    Args:
        gamma (float): focusing parameter (default: 2.0)
        alpha (float or None): balancing parameter (default: None)
        reduction (str): 'mean' or 'sum' (default: 'mean')
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, input, target):
        # We use binary cross-entropy with logits to compute the baseline loss for each sample.
        bce_loss = self.bce(input, target)
        # probas = torch.sigmoid(input)

        # We convert BCE_loss to pt, which is the model’s predicted probability for the true class. This becomes the key to calculating how “hard” each sample is.
        # if target = 1 then pt = proba
        # if target = 0 then pt = 1 - proba
        pt = torch.exp(-bce_loss) # exactly equivalent to:
        # pt = probas * target + (1 - probas) * (1 - target)

        # focal_weight amplifies the loss on harder samples,
        # gamma == 0 => no focal loss, just BCE
        focal_weight = (1 - pt) ** self.gamma
        # and alpha balances class importance, letting you decide how much priority to give minority classes.
        alpha_weight = 1
        if self.alpha is not None:
            #
            # Convention focal loss (Lin et al., 2017) :
            #   alpha est le poids appliqué aux exemples où target=1 (positifs, franchissement de seuil)
            #   1-alpha est le poids appliqué aux exemples où target=0 (négatifs, non franchissement)
            #
            # Donc :
            #   - Si alpha > 0.5, on donne PLUS de poids aux 1 (minoritaires)
            #   - Si alpha < 0.5, on donne PLUS de poids aux 0 (majoritaires)
            #
            # Pour CORAL, si les 1 sont rares (franchissement de seuil), il faut alpha > 0.5
            #
            alpha_weight = self.alpha * target + (1 - self.alpha) * (1 - target)
        loss = alpha_weight * focal_weight * bce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def plot_data(all_true, all_pred, train_loss, val_loss, val_accuracy, val_close_accuracy, val_mae_list, outputs_suffix):
    """Plot training and validation loss, accuracy, MAE, and confusion matrix in a 2x2 grid."""

    # Create a figure with 2x2 subplots
    plt.figure(figsize=(15, 12))

    # Plot 1: Training and Validation Loss
    plt.subplot(2, 2, 1)
    plot_shift = 20  # Shift to start plotting from epoch 20
    plt.plot(range(plot_shift, len(train_loss)), train_loss[plot_shift:], marker='o', label='Train Loss')
    plt.plot(range(plot_shift, len(val_loss)), val_loss[plot_shift:], marker='x', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()

    # Plot 2: Validation Accuracy (both exact and close)
    plt.subplot(2, 2, 2)
    plt.plot(val_accuracy, label='Exact Accuracy', color='green')
    plt.plot(val_close_accuracy, label='Close Accuracy (±1)', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid()

    # Plot 3: Validation MAE
    plt.subplot(2, 2, 3)
    plt.plot(val_mae_list, label='Validation MAE', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Validation Mean Absolute Error')
    plt.legend()
    plt.grid()

    # Plot 4: Confusion Matrix
    plt.subplot(2, 2, 4)
    cm = skmetrics.confusion_matrix(all_true, all_pred, labels=list(range(len(GRADE_DICT))))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(GRADE_DICT.keys()),
                yticklabels=list(GRADE_DICT.keys()))
    plt.xlabel('Predicted Grade')
    plt.ylabel('True Grade')
    plt.title('Confusion Matrix (Validation)')

    plt.tight_layout()
    plt.savefig(f"training_results-{outputs_suffix}.png", dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()


def train_model(dataset, hidden_size=128, batch_size=64, learning_rate=1e-3, weight_decay_factor=0.1, num_epochs=100, search_mode: bool = False, focal_gamma: float = 2.0, focal_alpha: float = None, early_stopping: bool = True, input_size=11) -> dict[str, float | dict[str, float]]:
    """
    Evaluate a set of hyperparameters and return key metrics for optimization.

    Returns:
        dict: Dictionary containing validation accuracy, loss, MAE, and best epoch
    """
    print(f"Training model with hyperparameters: hidden_size={hidden_size}, batch_size={batch_size}, learning_rate={learning_rate}, weight_decay_factor={weight_decay_factor}, num_epochs={num_epochs}")

    # Initialize model and dataset
    model = BoulderClassifier(input_size=input_size, hidden_size=hidden_size, num_classes=len(GRADE_DICT))
    model.to(device)

    train_loader, val_loader = initialize_dataloader(dataset, batch_size=batch_size)

    # Training setup

    # Define loss function
    loss_func = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)

    # weight decay is set to 10% of the learning rate.
    # This is a common practice in deep learning to prevent overfitting
    # and encourage generalization.
    # It helps to regularize the model by penalizing large weights.
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=learning_rate * weight_decay_factor)

    best_accuracy = 0.0
    best_mae = float('inf')  # MAE should be minimized
    best_close_accuracy = 0.0
    best_loss = float('inf')
    best_epoch = 0
    train_loss, val_loss, val_accuracy, val_close_accuracy, val_mae_list = [], [], [], [], []
    all_true = []
    all_pred = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        for batch_count, (boulders, grades) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(boulders)
            loss = loss_func(outputs, grades)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / (batch_count + 1)
        train_loss.append(avg_loss)

        # Validation phase
        model.eval()
        val_epoch_loss = 0.0
        correct = 0
        close_correct = 0
        total = 0
        mae_sum = 0.0
        all_true = []
        all_pred = []

        with torch.no_grad():
            for val_batch_count, (boulders, grades) in enumerate(val_loader):
                outputs = model(boulders)
                loss = loss_func(outputs, grades)
                val_epoch_loss += loss.item()

                preds = (torch.sigmoid(outputs) > 0.5).float()
                pred_idx = prediction2labelindex(preds.cpu().numpy())
                true_idx = prediction2labelindex(grades.cpu().numpy())

                # Exact accuracy (traditional)
                exact_matches = (pred_idx == true_idx).sum()
                correct += exact_matches

                # Close accuracy (±1 grade)
                close_matches = (np.abs(pred_idx - true_idx) <= 1).sum()
                close_correct += close_matches

                total += grades.size(0)
                mae_sum += np.abs(pred_idx - true_idx).sum()

                # For confusion matrix: convert ordinal output to class index
                all_true.extend(true_idx.tolist())
                all_pred.extend(pred_idx.tolist())

        avg_val_loss = val_epoch_loss / (val_batch_count + 1)
        exact_accuracy = 100.0 * correct / total if total > 0 else 0.0
        close_accuracy = 100.0 * close_correct / total if total > 0 else 0.0
        mae = mae_sum / total if total > 0 else 0.0
        val_loss.append(avg_val_loss)
        val_accuracy.append(exact_accuracy)
        val_close_accuracy.append(close_accuracy)
        val_mae_list.append(mae)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Valid Loss: {avg_val_loss:.4f}")
        print(f"    Exact Acc: {exact_accuracy:.2f}%, Close Acc (±1): {close_accuracy:.2f}%, MAE: {mae:.3f}")

        # Track best metrics - use MAE as primary metric (lower is better)
        if mae < best_mae:
            best_mae = mae
            best_accuracy = exact_accuracy
            best_close_accuracy = close_accuracy
            best_loss = avg_val_loss
            best_epoch = epoch + 1

        # Early stopping based on MAE improvement
        if epoch - best_epoch > 20 and early_stopping:
            print(f"Early stopping at epoch {epoch + 1} (no MAE improvement)")
            break
    if not search_mode:
        outputs_suffix = f"lr_{learning_rate}-epochs_{epoch+1}-hs_{hidden_size}-mae_{best_mae:.3f}-acc_{best_accuracy:.2f}"
        print(f"Training complete. Best MAE: {best_mae:.3f} with exact accuracy {best_accuracy:.2f}% and close accuracy {best_close_accuracy:.2f}% at epoch {best_epoch}")
        # Save the model
        prefix = "" if input_size == 11 else "ALLMOON-"
        model_path = f"{prefix}boulder_classifier-{outputs_suffix}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved as {model_path}")
        plot_data(all_true, all_pred, train_loss, val_loss, val_accuracy, val_close_accuracy, val_mae_list, outputs_suffix)
    return {
        'validation_accuracy': best_accuracy,
        'validation_close_accuracy': best_close_accuracy,
        'validation_loss': best_loss,
        'mae': best_mae,
        'best_epoch': best_epoch,
        'hidden_size': hidden_size,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'weight_decay_factor': weight_decay_factor,
        'focal_gamma': focal_gamma,
        'focal_alpha': focal_alpha
    }


def hyperparameter_search(dataset, input_size=11):
    """
    Example hyperparameter search function.
    You can use this as a starting point for grid search or random search.
    """
    # Define hyperparameter ranges
    hidden_sizes = [64, 128, 256, 512]
    batch_sizes = [32, 64, 128]
    learning_rates = [1e-5, 1e-4, 1e-3]
    # weight_decay_factors = [0, 0.01, 0.05, 0.1]
    # hidden_sizes = [512]
    # batch_sizes = [64]
    # learning_rates = [1e-3]
    weight_decay_factors = [0]
    focal_gammas = [0, 1.0, 2.0, 3.0, 4.0, 5.0]  # focus parameter for Focal Loss 1-5 is a common range 0 is no focal loss
    focal_alphas = [0.25, 0.5, 0.6, 0.75, 0.8]  # class weights for Focal Loss, 0.5 means no weighting < 0.5 means more weight on 1s (franchissement de seuil), > 0.5 means more weight on 0s (non franchissement de seuil)

    best_config = {}
    best_mae = float('inf')  # MAE should be minimized, not maximized
    results = []

    # Initialize CSV file with headers and read existing results
    csv_filename = "hyperparameter_results.csv" if input_size == 11 else f"hyperparameter_results_allmoon.csv"
    tested_configs = set()

    if os.path.exists(csv_filename) and os.path.getsize(csv_filename) > 0:
        print("Reading existing results from CSV...")
        with open(csv_filename, "r", newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Create config tuple for fast lookup
                config_tuple = (
                    int(row['hidden_size']),
                    int(row['batch_size']),
                    float(row['learning_rate']),
                    float(row['weight_decay_factor']),
                    float(row['focal_gamma']),
                    None if row['focal_alpha'] in ('', 'None') else float(row['focal_alpha'])
                )
                tested_configs.add(config_tuple)

                # Track best configuration from existing results
                mae = float(row['mae'])
                if mae < best_mae:
                    best_mae = mae
                    best_config = {
                        'hidden_size': int(row['hidden_size']),
                        'batch_size': int(row['batch_size']),
                        'learning_rate': float(row['learning_rate']),
                        'weight_decay_factor': float(row['weight_decay_factor']),
                        'focal_gamma': float(row['focal_gamma']),
                        'focal_alpha': None if row['focal_alpha'] in ('', 'None') else float(row['focal_alpha']),
                        'mae': mae,
                        'validation_accuracy': float(row['validation_accuracy']),
                        'validation_close_accuracy': float(row['validation_close_accuracy']),
                        'validation_loss': float(row['validation_loss']),
                        'best_epoch': int(row['best_epoch'])
                    }
        print(f"Found {len(tested_configs)} already tested configurations")
        if best_config:
            print(f"Current best MAE from existing results: {best_mae:.3f}")
    else:
        # Create new CSV file with headers
        with open(csv_filename, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "hidden_size", "batch_size", "learning_rate", "weight_decay_factor",
                "focal_gamma", "focal_alpha",
                "mae", "validation_accuracy", "validation_close_accuracy", "validation_loss", "best_epoch"
            ])

    print("Starting hyperparameter search...")
    print("Primary metric: MAE (lower is better)")
    print("Secondary metrics: Close accuracy (±1 grade), Exact accuracy")

    try:
        for hidden_size in hidden_sizes:
            for batch_size in batch_sizes:
                for learning_rate in learning_rates:
                    for weight_decay_factor in weight_decay_factors:
                        for focal_gamma in focal_gammas:
                            for focal_alpha in focal_alphas:
                                config_tuple = (hidden_size, batch_size, learning_rate, weight_decay_factor, focal_gamma, focal_alpha)
                                if config_tuple in tested_configs:
                                    print(f"Skipping already tested configuration: {config_tuple}")
                                    continue

                                config = {
                                    'hidden_size': hidden_size,
                                    'batch_size': batch_size,
                                    'learning_rate': learning_rate,
                                    'weight_decay_factor': weight_decay_factor,
                                    'focal_gamma': focal_gamma,
                                    'focal_alpha': focal_alpha
                                }
                                print(f"\nTesting configuration: {config}")
                                result = train_model(dataset, num_epochs=900, search_mode=True, **config)
                                results.append(result)

                                # Save result to CSV immediately after each configuration
                                with open(csv_filename, "a", newline='') as f:
                                    writer = csv.writer(f)
                                    writer.writerow([
                                        hidden_size,
                                        batch_size,
                                        learning_rate,
                                        weight_decay_factor,
                                        focal_gamma,
                                        focal_alpha,
                                        result['mae'],
                                        result['validation_accuracy'],
                                        result['validation_close_accuracy'],
                                        result['validation_loss'],
                                        result['best_epoch']
                                    ])

                                # Track best configuration by MAE (lower is better)
                                if result['mae'] < best_mae:
                                    best_mae = result['mae']
                                    best_config = result

                                print(f"MAE: {result['mae']:.3f}, Exact Acc: {result['validation_accuracy']:.2f}%, Close Acc (±1): {result['validation_close_accuracy']:.2f}%")
    except KeyboardInterrupt:
        print("\nRecherche interrompue par l'utilisateur (Ctrl-C).\nMeilleure configuration trouvée jusqu'ici :")
        print(f"Best MAE: {best_config['mae']:.3f}")
        print(f"Exact Accuracy: {best_config['validation_accuracy']:.2f}%, Close Accuracy: {best_config['validation_close_accuracy']:.2f}%")
        print(f"Hidden Size: {best_config['hidden_size']}, Batch Size: {best_config['batch_size']}")
        print(f"Learning Rate: {best_config['learning_rate']}, Weight Decay: {best_config['weight_decay_factor']}")
        print(f"Focal Gamma: {best_config['focal_gamma']}, Focal Alpha: {best_config['focal_alpha']}")
        return results, best_config

    print(f"\nBest configuration saved to {csv_filename}")
    print(f"Best MAE: {best_config['mae']:.3f}")
    print(f"Exact Accuracy: {best_config['validation_accuracy']:.2f}%, Close Accuracy: {best_config['validation_close_accuracy']:.2f}%")
    print(f"Hidden Size: {best_config['hidden_size']}, Batch Size: {best_config['batch_size']}")
    print(f"Learning Rate: {best_config['learning_rate']}, Weight Decay: {best_config['weight_decay_factor']}")
    print(f"Focal Gamma: {best_config['focal_gamma']}, Focal Alpha: {best_config['focal_alpha']}")

    return results, best_config


def loading_dataset(holds_data=None):
    print("Loading dataset...")
    dataset = commons.load_boulders_from_dataset()
    print(f"Loaded {len(dataset)} boulders from dataset, filtering...")
    filtered_dataset = filter_dataset(dataset)
    print(f"Filtered dataset now {len(filtered_dataset)} boulders, splitting into train (75%)/val (25%) sets...")
    dataset = BoulderDataset(filtered_dataset, {"2019": holds_data})
    return dataset


def initialize_dataloader(dataset, batch_size=128):
    train_size = int(0.75 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"Dataset initialized with {len(dataset)} samples. Train: {train_size}, Val: {val_size}")
    return train_loader, val_loader


def prediction2labelindex(pred: np.ndarray) -> int:
    """Convert ordinal predictions to class labels, e.g.

    [0.9, 0.1, 0.1, 0.1] -> 0
    [0.9, 0.9, 0.1, 0.1] -> 1
    [0.9, 0.9, 0.9, 0.1] -> 2
    etc.
    """
    pred = np.asarray(pred)
    if pred.ndim == 1:
        pred = pred.reshape(1, -1)
    return (pred > 0.5).cumprod(axis=1).sum(axis=1)


def prediction2probadist(threshold_probs: np.ndarray) -> dict[str, float]:
    """
    """
    # pred = np.asarray(pred)
    # if pred.ndim == 1:
    #     pred = pred.reshape(1, -1)
    # Convert CORAL threshold probabilities to class probabilities
    grade_probabilities = {}
    grade_keys = list(GRADE_DICT.keys())

    for i, grade in enumerate(grade_keys):
        if i == 0:
            # First grade: probability that no thresholds are passed
            prob = 1.0 - threshold_probs[0]
        elif i == len(grade_keys) - 1:
            # Last grade: probability that all thresholds are passed
            prob = threshold_probs[-1]
            for j in range(len(threshold_probs) - 1):
                prob *= threshold_probs[j]
        else:
            # Middle grades: probability that exactly i thresholds are passed
            # P(exactly i thresholds) = P(first i thresholds) * P(not (i+1)th threshold)
            prob = (1.0 - threshold_probs[i]) if i < len(threshold_probs) else 0.0
            for j in range(i):
                prob *= threshold_probs[j]

        grade_probabilities[grade] = float(prob)

    # Normalize probabilities to sum to 1
    total_prob = sum(grade_probabilities.values())
    if total_prob > 0:
        for grade in grade_probabilities:
            grade_probabilities[grade] /= total_prob

    return grade_probabilities


#####################
# Predicting Boulder Grade
######################

def predict_boulder_grade(boulder, model_path, input_size=11, version="2019"):
    """
    Given a boulder dict, loads the saved model and returns probabilities for each grade.

    Returns:
        dict: Dictionary mapping grade names to their probabilities
    """
    # Vectorize boulder
    holds_data = commons.load_holds_data(version)
    boulder_vector = BoulderDataset.vectorize_boulder(boulder, input_size, {"2019": holds_data})
    x = torch.tensor(boulder_vector, dtype=torch.float32).unsqueeze(0).to(device)  # shape (1, 14, input_size)

    # Extract hidden_size from model_path if present
    hidden_size = 128  # default
    match = re.search(r"-hs_(\d+)", model_path)
    if match:
        hidden_size = int(match.group(1))

    # Create model with correct hidden_size
    print(f"Loading model from {model_path} with hidden size {hidden_size}")
    model = BoulderClassifier(input_size=input_size, hidden_size=hidden_size, num_classes=len(GRADE_DICT))
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    with torch.no_grad():
        logits = model(x)
        threshold_probs = torch.sigmoid(logits).cpu().numpy()[0]  # Probabilities for each threshold
        grade_probabilities = prediction2probadist(threshold_probs)  # Convert to grade probabilities
    # Find the grade with highest probability
    # pred_grade = max(grade_probabilities.items(), key=lambda x: x[1])[0]
    # prob_percent = grade_probabilities[pred_grade] * 100
    return grade_probabilities

################
# Main function to run the training
################


def main(phase, boulder_json = None, model_path = None, boulder_object = None):
    if phase == "train":
        holds_data = commons.load_holds_data()
        dataset = loading_dataset(holds_data)
        train_model(dataset, hidden_size=512, batch_size=64, learning_rate=1e-3, weight_decay_factor=0, num_epochs=52, focal_gamma=5.0, focal_alpha=0.75, early_stopping=True)
    elif phase == "predict":
        boulder = boulder_object
        if boulder_json:
            boulder = json.load(boulder_json)
        pred_grade = predict_boulder_grade(boulder, model_path)
        if boulder_object:
            return pred_grade
        print(f"Predicted grades: {pred_grade}")
    elif phase == "search":
        holds_data = commons.load_holds_data()
        dataset = loading_dataset(holds_data)
        results, best_config = hyperparameter_search(dataset)


if __name__ == "__main__":
    main(*sys.argv[1:])
