# Importing Necessary packages
from flask import Flask, render_template, request, jsonify
import numpy as np
from fastai.vision import *
from fastbook import *

# from fastai.vision.widgets import *

import pickle
import io
import os


# path = Path('imagepath4')
# path = 1
# fns = get_image_files(path)


# Initialiazing flask app
app = Flask(__name__)

cwd = os.getcwd()
path = cwd + "/model"

print(path)

address = path + "/model.pkl"

# Loading  saved model
model = load_learner(address)

# Rendering index.html at / (looks in templates folder)
@app.route("/")
def index():
    return render_template("index.html")


# Getting data with POST Method
@app.route("/upload", methods=["POST"])
def upload():
    # try:
    # Getting img from POST
    file = request.files["user-img"].read()
    # Resizing img to 224 X 224 , This is the size on which model was trained
    #         img = open_image(io.BytesIO(file))
    img = PILImage.create(file)
    # Prediction using model
    prediction = model.predict(img)[0]

    # Getting Prediction ready to sent it to frontend
    response = {"result": str(prediction)}
    return jsonify(response)


### Routing option 1
@app.route("/hello")
def hello_world():
    return "hello this is a test"


# running app at localhost on port 8080
if __name__ == "__main__":

    app.run(host="0.0.0.0", port=8080)
