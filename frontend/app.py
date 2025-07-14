from flask import Flask
from flask import render_template, url_for
import moonboard.setter as setter
import moonboard.similar_boulder as sb
import moonboard.commons as commons
from unique_names_generator import get_random_name
from unique_names_generator.data import ADJECTIVES, ANIMALS, COLORS, COUNTRIES, LANGUAGES, NAMES, STAR_WARS
import random

app = Flask(__name__)


@app.route("/")
def hello_world():
    return render_template("index.html", title="Moonboard 2019 40°", image_url=url_for('static', filename='2019.png'))

@app.route("/generate")
def generate_boulder():
    boulder = setter.get_boulder()
    similar_boulders, score = sb.similar_boulders(boulder, sb.load_boulders_from_dataset(commons.DATASET_PATH))
    return {
        "boulder": ','.join(boulder),  # Convert holds to comma-separated string
        "score": f"{score*100:.2f}%",
        "similar": similar_boulders
    }

@app.route("/generate-name")
def generate_name():
    data_left = ADJECTIVES + COLORS + LANGUAGES
    data_right = ADJECTIVES + COLORS + ANIMALS + STAR_WARS
    name = get_random_name(combo=[data_left, data_right])
    print(f"Generated name: {name}")
    return {"name": name}