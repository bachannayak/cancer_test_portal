from flask import Flask,render_template,request
import pickle
import pandas as pd
import tensorflow as tf
import numpy as np
app = Flask('__name__')



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=["POST"])
def predict():
    model = tf.keras.models.load_model('my_model.h5')
    feature_value=[[float(x) for x in request.form.values()]]
    
    features=['mean radius', 'mean texture', 'mean perimeter', 'mean area',
       'mean smoothness', 'mean compactness', 'mean concavity',
       'mean concave points', 'mean symmetry', 'mean fractal dimension',
       'radius error', 'texture error', 'perimeter error', 'area error',
       'smoothness error', 'compactness error', 'concavity error',
       'concave points error', 'symmetry error', 'fractal dimension error',
       'worst radius', 'worst texture', 'worst perimeter', 'worst area',
       'worst smoothness', 'worst compactness', 'worst concavity',
       'worst concave points', 'worst symmetry', 'worst fractal dimension']

    
    feature_final=np.array(feature_value).reshape(-1,30)
    
    prediction=model.predict(feature_final)
    if (prediction[0][0] > 0.5):
        
        result="tumors are non-cancerous"
    else:
        result='tumors are cancerous'
    return render_template('index.html',prediction_text='The {}'.format(result))

if(__name__=='__main__'):
    app.run(debug=True)

