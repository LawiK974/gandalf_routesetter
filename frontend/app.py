import moonboard.beta as beta
from flask import Flask
from flask import render_template, url_for, request
import moonboard.setter as setter
import moonboard.similar_boulder as sb
import moonboard.commons as commons
import moonboard.grader as grader
import moonboard.supergrader as supergrader
from unique_names_generator import get_random_name
from unique_names_generator.data import ADJECTIVES, ANIMALS, COLORS, COUNTRIES, LANGUAGES, NAMES, STAR_WARS
import os
import traceback
import json
import flask_limiter
import flask_limiter.util as flutil

app = Flask(__name__)

limiter = flask_limiter.Limiter(
    key_func=flutil.get_remote_address,
    app=app,
    default_limits=["10 per minute"],
)

@app.route("/<version>")
@app.route("/")
@limiter.exempt
def hello_world(version="2019"):
    return render_template("index.html", title=f"Moonboard {version} 40°", image_url=url_for('static', filename=f"{version}.png"), version=version)

@app.route("/generate")
@limiter.limit("10 per minute")
def generate_boulder():
    span = int(request.args.get('span', 170))
    hold_types = request.args.get('hold_types')
    version=request.args.get('version', '2019')
    if hold_types:
        hold_types = hold_types.split(',')
    else:
        hold_types = None
    try:
        boulder = setter.get_boulder(span=span, hold_types=hold_types, version=version)
        filtered_dataset = None
        if os.path.exists(commons.FILTERED_DATASET_PATH[version]):
            # Check if the generated boulder is in the filtered dataset
            with open(commons.FILTERED_DATASET_PATH[version], 'r') as f:
                filtered_dataset = json.load(f)
        similar_boulders, score = sb.similar_boulders(boulder, filtered_dataset or sb.load_boulders_from_dataset(commons.DATASET_PATH[version]))
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


@app.route("/get-beta/<version>")
@app.route("/get-beta/")
@limiter.limit("1 per minute")
def get_beta(version="2019"):
    span = int(request.args.get('span', 170))
    boulder = json.loads(request.args.get('boulder'))
    if not boulder:
        return {"error": "Missing boulder parameter."}, 400
    start = boulder[:2] if int(boulder[1][1:]) <= 6 else boulder[:1]
    boulder_dict = {
        "holds": boulder,
        "start": start,
        "end": [boulder[-1]],
        "version": version 
    }
    try:
        # beta.main attend une string de type '["A1", "B2", ...]'
        best_betas = beta.main(span, boulder_dict = boulder_dict, version=version)
        return {"betas": best_betas, "error": None}
    except Exception as e:
        # raise e  # Re-raise the exception to be caught by the Flask error handler
        traceback.print_exc()
        return {"betas": [], "error": str(e)}, 500
    


@app.route("/predict-grade/<version>", methods=["POST"])
@app.route("/predict-grade/", methods=["POST"])
@limiter.limit("10 per minute")
def predict_grade(version="2019"):
    data = request.get_json()
    boulder = data.get("boulder")
    if not boulder:
        return {"error": "Missing boulder parameter."}, 400
    # Load model path (assume latest or best model)
    model_dir = os.path.join(os.path.dirname(__file__), '../')
    simple_model = {
        "2019": "2019-boulder_classifier-lr_0.001-epochs_76-hs_512-mae_0.244-acc_84.04.pth",
        "2016": "2016-boulder_classifier-lr_0.001-epochs_42-hs_512-mae_0.243-acc_84.89.pth",
        "2017": "2017-boulder_classifier-lr_0.001-epochs_53-hs_512-mae_0.125-acc_92.27.pth",
    }[version]
    complex_model = "ALLMOON-boulder_classifier-lr_0.001-epochs_48-hs_512-mae_0.303-acc_81.47.pth"
    simple_model = os.path.join(model_dir, simple_model)
    complex_model = os.path.join(model_dir, complex_model)
    # Build a fake boulder dict for prediction
    if not os.path.exists(simple_model):
        return {"error": "No trained model found."}, 500
    start = boulder[:2] if int(boulder[1][1:]) <= 6 else boulder[:1]
    boulder_dict = {
        "holds": boulder,
        "start": start,
        "end": [boulder[-1]],
        "version": version  # Default version, can be changed if needed
    }
    try:
        simple_pred = grader.main("predict", boulder_object=boulder_dict, model_path=simple_model, version=version)
        complex_pred_grade = supergrader.main("predict", boulder_object=boulder_dict, model_path=complex_model)
        return {
            "simple_pred": simple_pred,
            "complex_pred": complex_pred_grade,
            "error": None
        }
    except Exception as e:
        traceback.print_exc()
        return {"grade": '', "probability": '', "error": str(e)}, 500
