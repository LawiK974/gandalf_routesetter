import moonboard.beta as beta
from flask import Flask
from flask import render_template, url_for, request
import moonboard.setter as setter
import moonboard.similar_boulder as sb
import moonboard.commons as commons
import moonboard.grader as grader
from unique_names_generator import get_random_name
from unique_names_generator.data import ADJECTIVES, ANIMALS, COLORS, COUNTRIES, LANGUAGES, NAMES, STAR_WARS
import os
import traceback

app = Flask(__name__)


@app.route("/")
def hello_world():
    return render_template("index.html", title="Moonboard 2019 40°", image_url=url_for('static', filename='2019.png'))

@app.route("/generate")
def generate_boulder():
    span = int(request.args.get('span', 170))
    hold_types = request.args.get('hold_types')
    if hold_types:
        hold_types = hold_types.split(',')
    else:
        hold_types = None
    try:
        boulder = setter.get_boulder(span=span, hold_types=hold_types)
        similar_boulders, score = sb.similar_boulders(boulder, sb.load_boulders_from_dataset(commons.DATASET_PATH['2019']))
        return {
            "boulder": ','.join(boulder),  # Convert holds to comma-separated string
            "score": f"{score*100:.2f}%",
            "similar": similar_boulders,
            "error": None
        }
    except Exception as e:
        traceback.print_exc()
        return {
            "boulder": '',
            "score": '',
            "similar": [],
            "error": str(e)
        }, 500


@app.route("/generate-name")
def generate_name():
    data_left = ADJECTIVES + COLORS + LANGUAGES
    data_right = ADJECTIVES + COLORS + ANIMALS + STAR_WARS
    name = get_random_name(combo=[data_left, data_right])
    print(f"Generated name: {name}")
    return {"name": name}


@app.route("/get-beta")
def get_beta():
    span = int(request.args.get('span', 170))
    boulder = request.args.get('boulder')
    if not boulder:
        return {"error": "Missing boulder parameter."}, 400
    try:
        # beta.main attend une string de type '["A1", "B2", ...]'
        best_betas = beta.main(span, boulder)
        return {"betas": best_betas, "error": None}
    except Exception as e:
        # raise e  # Re-raise the exception to be caught by the Flask error handler
        traceback.print_exc()
        return {"betas": [], "error": str(e)}, 500
    


@app.route("/predict-grade", methods=["POST"])
def predict_grade():
    data = request.get_json()
    boulder = data.get("boulder")
    if not boulder:
        return {"error": "Missing boulder parameter."}, 400
    # Load model path (assume latest or best model)
    model_dir = os.path.join(os.path.dirname(__file__), '../')
    # Find a .pth file
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    if not model_files:
        return {"error": "No trained model found."}, 500
    simple_model = "boulder_classifier-lr_0.001-epochs_52-hs_512-mae_0.288-acc_82.08.pth"
    # complex_model = "beta_classifier-lr_0.001-epochs_250-hs_512-acc_81.43-loss_0.160.pth"
    simple_model = os.path.join(model_dir, simple_model)
    # complex_model = os.path.join(model_dir, complex_model)
    # Build a fake boulder dict for prediction
    start = boulder[:2] if int(boulder[2][1:]) < 6 else boulder[:1]
    boulder_dict = {
        "holds": boulder,
        "start": start,
        "end": [boulder[-1]],
        "version": "2019"  # Default version, can be changed if needed
    }
    try:
        simple_pred = grader.main("predict", boulder_object=boulder_dict, model_path=simple_model)
        # complex_pred_grade, complex_prob_percent = supergrader.main("predict", boulder_object=boulder_dict, model_path=complex_model)
        return {
            "simple_pred": simple_pred,
            "complex_pred": "N/A",
            "error": None
        }
    except Exception as e:
        traceback.print_exc()
        return {"grade": '', "probability": '', "error": str(e)}, 500
