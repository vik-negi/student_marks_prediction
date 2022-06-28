# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib
import os

app = Flask(__name__)
# this
model = joblib.load("students_marks_predictor.pkl")

df = pd.DataFrame()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    global df
    
    input_features = [float(x) for x in request.form.values()]
    features_value = np.array(input_features)
    
    #validate input hours
    if input_features[0] <0 or input_features[0] >24:
        return render_template('index.html', prediction_text='Please enter valid hours between 1 to 24 if you live on the Earth')
    elif input_features[0]<0.2:
        return render_template('index.html', prediction_text='You will get 0% marks, when you do study {} hours per day'.format(features_value[0]))
    elif input_features[0]<=0.5:
        return render_template('index.html', prediction_text='You will get 45% marks, when you do study {} hours per day'.format(features_value[0]))
    elif input_features[0]<=0.8:
        return render_template('index.html', prediction_text='You will get 50% marks, when you do study {} hours per day'.format(features_value[0]))
    elif input_features[0]>12:
        return render_template('index.html', prediction_text='You will get more than 98% marks, if you study 12 or more than 12 hours per day ')

    output = model.predict([features_value])[0][0].round(2)

    # input and predicted value store in df then save in csv file
    df= pd.concat([df,pd.DataFrame({'Study Hours':input_features,'Predicted Output':[output]})],ignore_index=True)
    print(df)   
    df.to_csv('smp_data_from_app.csv')

    return render_template('index.html', prediction_text='You will get {}% marks, when you do study {} hours per day '.format(output,(features_value[0])))

port = int(os.environ.get('PORT', 5000))
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=port, debug=True)
    