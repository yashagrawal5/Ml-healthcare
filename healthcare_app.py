import jsonify
import requests
import pickle
import numpy as np
import sys
import os
import re
import sklearn
from flask import Flask, render_template, url_for, flash, redirect, request, send_from_directory
from sklearn.preprocessing import StandardScaler
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from werkzeug.utils import secure_filename

app = Flask(__name__)

model_heartdisease = pickle.load(open('heartdisease.pkl', 'rb'))
model_liverdisease = pickle.load(open('liverdisease.pkl', 'rb'))
model_cancer = pickle.load(open('breastcancer.pkl', 'rb'))



@app.route('/',methods=['GET'])
@app.route('/home',methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/heartdisease', methods=['GET','POST'])
def heartdisease():
    if request.method == 'POST':
        Age=int(request.form['age'])
        Gender=int(request.form['sex'])
        ChestPain= int(request.form['cp'])
        BloodPressure= int(request.form['trestbps'])
        ElectrocardiographicResults= int(request.form['restecg'])
        MaxHeartRate= int(request.form['thalach'])
        ExerciseInducedAngina= int(request.form['exang'])
        STdepression= float(request.form['oldpeak'])
        ExercisePeakSlope= int(request.form['slope'])
        MajorVesselsNo= int(request.form['ca'])
        Thalassemia=int(request.form['thal'])
        prediction=model_heartdisease.predict([[Age, Gender, ChestPain, BloodPressure, ElectrocardiographicResults, MaxHeartRate, ExerciseInducedAngina, STdepression, ExercisePeakSlope, MajorVesselsNo, Thalassemia]])
        if prediction==1:
            return render_template('heartdisease_prediction.html', prediction_text="Oops! You seem to have a Heart Disease.", title='Heart Disease')
        else:
            return render_template('heartdisease_prediction.html', prediction_text="Great! You don't have any Heart Disease.", title='Heart Disease')
    else:
        return render_template('heartdisease.html', title='Heart Disease')

    
@app.route('/liverdisease', methods=['GET','POST'])
def liverdisease():
    if request.method == 'POST':
        Age=int(request.form['Age'])
        Gender=int(request.form['Gender'])
        Total_Bilirubin= float(request.form['Total_Bilirubin'])
        Direct_Bilirubin= float(request.form['Direct_Bilirubin'])
        Alkaline_Phosphotase= int(request.form['Alkaline_Phosphotase'])
        Alamine_Aminotransferase= int(request.form['Alamine_Aminotransferase'])
        Aspartate_Aminotransferase= int(request.form['Aspartate_Aminotransferase'])
        Total_Protiens= float(request.form['Total_Protiens'])
        Albumin= float(request.form['Albumin'])
        Albumin_and_Globulin_Ratio= float(request.form['Albumin_and_Globulin_Ratio'])
        prediction=model_liverdisease.predict([[Age, Gender, Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphotase, Alamine_Aminotransferase, Aspartate_Aminotransferase, Total_Protiens, Albumin, Albumin_and_Globulin_Ratio]])
        if prediction==1:
            return render_template('liverdisease_prediction.html', prediction_text="Oops! You seem to have Liver Disease.", title='Liver Disease')
        else:
            return render_template('liverdisease_prediction.html', prediction_text="Great! You don't have any Liver Disease.", title='Liver Disease')
    else:
        return render_template('liverdisease.html', title='Liver Disease')

@app.route('/breastcancer', methods=['GET','POST'])
def breastcancer():
    if request.method == 'POST':
        texture_mean = float(request.form['texture_mean'])
        perimeter_mean = float(request.form['perimeter_mean'])
        smoothness_mean = float(request.form['smoothness_mean'])
        compactness_mean = float(request.form['compactness_mean'])
        concavity_mean = float(request.form['concavity_mean'])
        concave_points_mean = float(request.form['concave_points_mean'])
        symmetry_mean = float(request.form['symmetry_mean'])
        radius_se = float(request.form['radius_se'])
        compactness_se = float(request.form['compactness_se'])
        concavity_se = float(request.form['concavity_se'])
        concave_points_se = float(request.form['concave_points_se'])
        texture_worst = float(request.form['texture_worst'])
        smoothness_worst = float(request.form['smoothness_worst'])
        compactness_worst = float(request.form['compactness_worst'])
        concavity_worst = float(request.form['concavity_worst'])
        concave_points_worst = float(request.form['concave_points_worst'])
        symmetry_worst = float(request.form['symmetry_worst'])
        fractal_dimension_worst = float(request.form['fractal_dimension_worst'])
        prediction=model_cancer.predict([[texture_mean, perimeter_mean, smoothness_mean, compactness_mean,
           concavity_mean, concave_points_mean, symmetry_mean, radius_se,
           compactness_se, concavity_se, concave_points_se, texture_worst,
           smoothness_worst, compactness_worst, concavity_worst,
           concave_points_worst, symmetry_worst, fractal_dimension_worst]])
        if prediction==1:
            return render_template('cancer_prediction.html', prediction_text="Oops! The tumor is malignant.", title='Breast Cancer')
        else:
            return render_template('cancer_prediction.html', prediction_text="Great! The tumor is benign.", title='Breast Cancer')
    else:
        return render_template('cancer.html',title='Breast Cancer')


def pneumonia_predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    
    x=image.img_to_array(img)
    x=x/255
    x=np.expand_dims(x, axis=0)
    preds = model_pneumonia.predict(x)
    return preds





@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory('uploads', filename)

if __name__=='__main__':
	app.debug =True
	app.run()

