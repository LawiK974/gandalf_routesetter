#!/usr/bin/python
#-*- coding: utf-8 -*-

import numpy as np
from torch import nn
import torch
# from beta import possible_betas, best_betas
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import sys
import matplotlib.pyplot as plt
import json
import sklearn.metrics as skmetrics
import seaborn as sns
try:
    import commons
except:
    from . import commons
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##########################################
# Constants and Mappings
##########################################
# check https://towardsdatascience.com/how-to-perform-ordinal-regression-classification-in-pytorch-361a2a095a99/
# for why this encoding is used (CORAL method)
# The grades are encoded as a binary vector where each position corresponds to a threshold passed.
GRADE_DICT = {
    "6A+":[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "6B": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "6B+":[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "6C": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "6C+":[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "7A": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "7A+":[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    "7B": [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    "7B+":[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    "7C": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    "7C+":[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    "8A": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    "8A+":[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    "8B": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    "8B+":[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
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
        and (
            boulder['setter'] in list_benchmark_setters # benchmark setters
            or (
                boulder["userGrade"] is not None # user grade is set
                and boulder["grade"] != "8B+" # ignore 8B+ grades
                and boulder["userGrade"] != "8B+" # ignore 8B+ grades
                and not has_too_much_holds_together(boulder['holds'])
                and boulder["rating"] >= 3
            )
        )
    ), dataset))
    return filtered_dataset

# def get_betas_dataset(dataset, holds_data=None):
#     betas_dataset = []
#     for idx,boulder in enumerate(dataset):
#         betas = possible_betas(boulder["holds"], holds_data, span=175)
#         best_beta = best_betas(betas, max_betas=1)
#         if best_beta:
#             betas_dataset.append({
#                 **boulder,
#                 "beta": best_beta[0]
#             })
#         print(f"Processed {idx+1}/{len(dataset)} boulders for betas.")
#     # raise ValueError("Beta dataset generation completed.")
#     return betas_dataset

class BoulderDataset(Dataset):
    def __init__(self, dataset, holds_data=None):
        self.data, self.labels = [], []
        for boulder in dataset:
            boulder_vector = self.vectorize_boulder(boulder, holds_data)
            label_vector = GRADE_DICT[boulder["userGrade"] or boulder['grade']][1:]  # remove the first element (0) to match the output size
            self.data.append(torch.tensor(boulder_vector, dtype=torch.float32).to(device))
            self.labels.append(torch.tensor(label_vector, dtype=torch.float32).to(device))

    @staticmethod
    def vectorize_boulder(boulder, holds_data=None):
        """Convertit une beta en vecteur de difficulté des prises utilisées."""
        vector = np.zeros((14, 8))  # Maximum number of moves in a beta (app limit 14 holds )
        for idx, hold_name in enumerate(boulder["holds"]):
            hold_type = holds_data[hold_name]['type']
            hold_texture = holds_data[hold_name]['texture']
            can_match = holds_data[hold_name]['can_match']
            orientation = holds_data[hold_name]['orientation']
            is_start = hold_name in boulder["start"]
            is_end = hold_name in boulder["end"]
            hold_pos = commons.hold_name_to_pos(hold_name)
            vector[idx] = [
                int(hold_pos[0]),
                int(hold_pos[1]),
                int(is_start),
                int(is_end),
                int(HOLD_TYPE_DICT[hold_type]),
                int(HOLD_TEXTURE_DICT[hold_texture]),
                int(ORIENTATION_DICT[orientation]),
                int(CAN_MATCH_DICT[can_match]),
            ]
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
        out = self.fc(out[:, -1, :]) # Take the last time step of the lstm
        return out


def ordinal_regression(predictions: list[list[float]], targets: list[float]):
    """Ordinal regression with encoding as in https://arxiv.org/pdf/0704.1028.pdf"""

    return nn.MSELoss(reduction='none')(predictions, targets).sum(axis=1)


def train_model(model: BoulderClassifier, loaders: tuple[DataLoader], num_epochs=100, learning_rate=1e-3):
    train_loader, val_loader = loaders
    # Define loss function
    # Using BCEWithLogitsLoss for binary classification with logits output
    # This is suitable for ordinal regression where we treat each threshold as a binary classification problem.
    # It computes the binary cross-entropy loss between the predicted logits and the target labels.
    # It combines a sigmoid layer and the binary cross-entropy loss in one single class.
    # This is more numerically stable than using a plain Sigmoid followed by a BCELoss loss.
    # It is used for multi-label classification problems where each class is independent.
    loss_func = nn.BCEWithLogitsLoss()

    # loss_func = ordinal_regression
    
    # weight decay is set to 10% of the learning rate.
    # This is a common practice in deep learning to prevent overfitting
    # and encourage generalization.
    # It helps to regularize the model by penalizing large weights.
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)

    train_loss, val_loss = [], []
    for epoch in range(num_epochs):
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
        total = 0
        all_true = []
        all_pred = []
        with torch.no_grad():
            for val_batch_count, (boulders, grades) in enumerate(val_loader):
                outputs = model(boulders)
                loss = loss_func(outputs, grades)
                val_epoch_loss += loss.item()
                # Compute accuracy: convert sigmoid(outputs) > 0.5 to binary, then sum all correct full matches
                preds = (torch.sigmoid(outputs) > 0.5).float()
                # matches = (preds == grades).all(dim=1)
                pred_idx = prediction2labelindex(preds.cpu().numpy()) 
                true_idx = prediction2labelindex(grades.cpu().numpy())
                matches = (pred_idx == true_idx)
                correct += matches.sum().item()
                total += grades.size(0)
                # For confusion matrix: convert ordinal output to class index
                all_true.extend(true_idx.tolist())
                all_pred.extend(pred_idx.tolist())
        avg_val_loss = val_epoch_loss / (val_batch_count + 1)
        val_loss.append(avg_val_loss)
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Valid Loss: {avg_val_loss:.4f}, Valid Acc: {accuracy:.2f}%")

    # Plot confusion matrix
    cm = skmetrics.confusion_matrix(all_true, all_pred, labels=list(range(len(GRADE_DICT))))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(GRADE_DICT.keys()), yticklabels=list(GRADE_DICT.keys()))
    plt.xlabel('Predicted Grade')
    plt.ylabel('True Grade')
    plt.title('Confusion Matrix (Validation)')
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_val_lr_{learning_rate}_BCE_Adam.png")
    # plt.show()
    plt.close()

    # Plot loss evolution
    plt.figure()
    plt.plot(range(1, num_epochs+1), train_loss, marker='o', label='Train Loss')
    plt.plot(range(1, num_epochs+1), val_loss, marker='x', label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Evolution')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"training_val_loss_lr_{learning_rate}_BCE_Adam.png")
    # plt.show()
    plt.close()

    # Save the model
    model_path = f"beta_classifier_lr_{learning_rate}_{avg_loss:.4f}_Adam_BCE.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved as {model_path}")

    return train_loss, val_loss

def initialize_dataset(batch_size=128, holds_data=None):
    print("Loading dataset...")
    dataset = commons.load_boulders_from_dataset()
    print(f"Loaded {len(dataset)} boulders from dataset, filtering...")
    filtered_dataset = filter_dataset(dataset)
    print(f"Filtered dataset now {len(filtered_dataset)} boulders, splitting into train (75%)/val (25%) sets...")
    dataset = BoulderDataset(filtered_dataset, holds_data)
    train_size = int(0.75 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    print(f"Dataset initialized with {len(dataset)} samples. Train: {train_size}, Val: {val_size}")
    return train_loader, val_loader

def prediction2labelindex(pred: np.ndarray) -> int:
    """Convert ordinal predictions to class labels, e.g.
    
    [0.9, 0.1, 0.1, 0.1] -> 0
    [0.9, 0.9, 0.1, 0.1] -> 1
    [0.9, 0.9, 0.9, 0.1] -> 2
    etc.
    """
    return (pred > 0.5).cumprod(axis=1).sum(axis=1)


#####################
# Predicting Boulder Grade
######################

def predict_boulder_grade(model: BoulderClassifier, boulder, model_path, holds_data=None):
    """
    Given a boulder dict, loads the saved model and returns the predicted grade string.
    """
    # Vectorize boulder
    boulder_vector = BoulderDataset.vectorize_boulder(boulder, holds_data)
    x = torch.tensor(boulder_vector, dtype=torch.float32).unsqueeze(0).to(device)  # shape (1, 14, 8)
    # Load model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        # CORAL: find the number of thresholds passed (i.e., how many outputs > 0.5)
        pred_idx = prediction2labelindex(probs)
        # Map index to grade
        grade_keys = list(GRADE_DICT.keys())
        pred_grade = grade_keys[pred_idx]
        # Probability for the predicted grade: min(prob) of thresholds passed,
        # or 1-prob of first not passed
        if pred_idx == 0:
            prob = 1.0 - probs[0]
        elif pred_idx == len(probs):
            prob = probs[-1]
        else:
            prob = min(probs[:pred_idx])
        prob_percent = float(prob) * 100
    return pred_grade, prob_percent

################
# Main function to run the training
################

def main(phase, boulder_json = None, model_path = None):
    model = BoulderClassifier(input_size=8, hidden_size=128, num_classes=15)
    model.to(device)
    holds_data = commons.load_holds_data()
    if phase == "train":
        dataloader = initialize_dataset(batch_size=128, holds_data=holds_data)
        train_model(model, dataloader, num_epochs=100, learning_rate=1e-3)     
    elif phase == "predict":   
        if len(sys.argv) < 3:
            print("Usage: python grader.py predict <boulder_json> <model_path>")
            return
        boulder = json.load(boulder_json)
        pred_grade, prob_percent = predict_boulder_grade(model, boulder, model_path, holds_data)
        print(f"Predicted grade: {pred_grade}, Probability: {prob_percent:.2f}%")

if __name__ == "__main__":
    main(*sys.argv[1:])