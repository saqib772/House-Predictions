import pandas as pd
from flask import  Flask, render_template, request
import  pandas
app= Flask(__name__)

data=pd.read_csv('cleanHousedata.csv')
import  pickle
pipe=pickle.load(open("lassoModel.pkl",'rb'))

@app.route('/')

def index():
    locations= sorted(data['location'].unique())
    return  render_template('index.html',locations)

import  numpy as np
@app.route('/predict',methods=['POST'])
def predict():
    location=request.form.get('location')
    bhk=request.form.get('BHK')
    bath=request.form.get('bathrooms')
    sqft=request.form.get('square-feet')

    input=pd.DataFrame([[location,sqft,bath,bhk]],columns=['location','total_sqft','bath','bhk',])
    prediction= pipe.predict(input)[0] *1e5

    return str(np.round(prediction,2 ))
if __name__== "__main__":
    app.run(debug=True,port=5001)

