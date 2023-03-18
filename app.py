import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect, session, flash
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Flask SQLAlchemy
from flask_sqlalchemy import SQLAlchemy

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Preprocessing utilities
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Model building
from keras import layers
from keras.optimizers import Adam
from keras.models import Sequential
from keras.applications import DenseNet121
from keras.callbacks import Callback, ModelCheckpoint


from PIL import Image
from models.model import build_model, preprocess_image

# Some utilites
import numpy as np
from utils import base64_to_pil
# Creating a new Flask Web application.
app = Flask(__name__)

# Configure the database path
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'mysecretkey'
db = SQLAlchemy(app)


# Define the User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)


# Model saved with Keras model.save()
MODEL_PATH = './models/pretrained/model.h5'

# Loading trained model
model = build_model()
model.load_weights(MODEL_PATH)
print('Model loaded. Start serving...')


def model_predict(img, model):
    """
    Classify the severity of DR of image using pre-trained CNN model.

    Keyword arguments:
    img -- the retinal image to be classified
    model -- the pretrained CNN model used for prediction

    Predicted rating of severity of diabetic retinopathy on a scale of 0 to 4:

    0 - No DR
    1 - Mild
    2 - Moderate
    3 - Severe
    4 - Proliferative DR

    """
    
    ## Preprocessing the image
    x_val = np.empty((1, 224, 224, 3), dtype=np.uint8)
    img = img.resize((224,) * 2, resample=Image.LANCZOS)
    x_val[0, :, :, :] = img

    preds = model.predict(x_val)
    return preds


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        session.pop('logged_in', None)
        return redirect(url_for('index'))
    elif 'logged_in' in session:
        return render_template('index.html')
    else:
        return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        # Get the form data
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        # Check if the user already exists in the database
        if User.query.filter_by(username=username).first() or User.query.filter_by(email=email).first():
            return "User already exists"

        # Create a new user object
        user = User(username=username, email=email, password=password)

        # Add the user to the database
        db.session.add(user)
        db.session.commit()

        # Redirect to the login page
        return redirect(url_for('login'))

    # Render the signup page
    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # perform login authentication and set session
        session['logged_in'] = True
        return redirect(url_for('index'))
    else:
        return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.')
    return redirect(url_for('login'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Save the image to ./uploads
        # img.save("./uploads/image.png")

        # Make prediction on the image
        preds = model_predict(img, model)

        # Process result to find probability and class of prediction
        pred_proba = "{:.3f}".format(np.amax(preds))    # Max probability
        pred_class = np.argmax(np.squeeze(preds))

        diagnosis = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

        result = diagnosis[pred_class]               # Convert to string
        
        # Serialize the result
        return jsonify(result=result, probability=pred_proba)

    return None



if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
