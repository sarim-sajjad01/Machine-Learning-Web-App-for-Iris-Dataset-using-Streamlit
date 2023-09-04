import streamlit as st
import pandas as pd
import pickle as pk
from sklearn import datasets
import numpy as np


iris = datasets.load_iris()

st.write("""
# Iris Flower Prediction Application

This app predicts the **Iris flower** type!
         """)

st.sidebar.header('User Input Parameter')

def userInputFeatures():
    sepalLength = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepalWidth = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petalLength = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petalWidth = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepalLength,
            'sepal_width': sepalWidth,
            'petal_length': petalLength,
            'petal_width': petalWidth}
    features = pd.DataFrame(data, index=[0])
    return features

def Predistions(inputData):
    model = pk.load(open("model/model.pkl", "rb"))

    prediction = model.predict(inputData)
    prediction_proba = model.predict_proba(inputData)

    return prediction, prediction_proba

df = userInputFeatures()

GIFImage = 'https://1.bp.blogspot.com/-nUNmjUY0kwo/XIGaggwn58I/AAAAAAAAD4A/tA3S0Tgu5N4rbjlaNEbr_I5GuHbQwHyRgCEwYBhgL/s1600/image6.gif'
st.sidebar.image(GIFImage, caption="Face-Mesh", use_column_width=True)

st.subheader('User Input parameters')
st.write(df)

prediction, predictionProb = Predistions(df)

st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediction])
#st.write(prediction)

st.subheader('Prediction Probability')
st.write(predictionProb)

st.markdown(
        '''
            # About Me \n
              Hey! this is Engineer **Zia Ur Rehman**.

              If you are interested in building more **AI, Computer Vision and Machine Learning** projects, then visit my GitHub account. You can find a lot of projects with python code.

              - [GitHub](https://github.com/ZiaUrRehman-bit)
              - [LinkedIn](https://www.linkedin.com/in/zia-ur-rehman-217a6212b/) 
              - [Curriculum vitae](https://github.com/ZiaUrRehman-bit/ZiaUrRehman-bit/blob/main/A2023Updated.pdf)
              
              For any query, You can email me.
              *Email:* engrziaurrehman.kicsit@gmail.com
        '''
    )