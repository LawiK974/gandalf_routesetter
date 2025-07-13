from flask import Flask
from flask import render_template, url_for
import moonboard.setter as setter
import moonboard.similar_boulder as sb
import moonboard.commons as commons

app = Flask(__name__)


@app.route("/")
def hello_world():
    return render_template("index.html")

@app.route("/generate")
def generate_boulder():
    boulder = setter.get_boulder()
    similar_boulder, score = sb.similar_boulders(boulder, sb.load_boulders_from_dataset(commons.DATASET_PATH))
    return {
        "boulder": ','.join(boulder),  # Convert holds to comma-separated string
        "similar": f"Most similar Boulder: {similar_boulder}, Score: {score*100:.2f}%"
    }