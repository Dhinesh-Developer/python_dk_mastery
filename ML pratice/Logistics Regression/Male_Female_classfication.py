import pandas as pd
import streamlit as st
from pickle import load

st.title('Model Deployment: Logistic Regression')

st.sidebar.header('User Input Parameters')

def user_input_features():
    height = st.sidebar.number_input("Height (cm)")
    weight = st.sidebar.number_input("Weight (kg)")
    
    data = {'height': height, 'weight': weight}  # match training data
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.sidebar.write('User Input parameters')
st.write(df)

# Load model
loaded_model = load(open('gender_classfication_intelligence.pkl', 'rb'))

# Prediction
prediction_proba = loaded_model.predict_proba(df)
prediction = loaded_model.predict(df)

st.sidebar.write('Predicted Result')

# Show result
st.write('Male' if prediction[0] == 0 else 'Female')

st.subheader('Prediction Probability')
st.write(prediction_proba)