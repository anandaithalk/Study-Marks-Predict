# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

model = joblib.load("marks-predict.pkl")

df = pd.DataFrame()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['Get', 'POST'])
def predict():
    global df
    
    input_features = [int(x) for x in request.form.values()]
    features_value = np.array(input_features)
    
    
    
    # Validating input hours
    
    if input_features[0] <1 or input_features[0] >24:
        return render_template('index.html', prediction_text='Please enter valid hours between 1 to 24 hours')
        

    output = model.predict([features_value])[0][0].round(2)

    
    # Inputting and predicteding value store in df then save in csv file
    
    df= pd.concat([df,pd.DataFrame({'Study Hours':input_features,
                                    'Predicted Output':[output]})],ignore_index=True)
    print(df)   
    df.to_csv('smp_data_from_app.csv')

    return render_template('index.html', 
                           prediction_text='If you study for [{}] hours, you will achieve [{}]% marks'.format(input_features[0],output))
                           


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
    