import streamlit as st
import pandas as pd
from joblib import load
import dill

# Load the pretrained model
with open('pipeline1.pkl', 'rb') as file:
    model = dill.load(file)

feature_dict = load('feature_dict.pkl')

# Function to predict churn
def predict_mushroom(data):
    prediction = model.predict(data)
    return prediction


st.title('Mushroom Edible Predictor')

# Display features
st.subheader('Features')
feature_input = feature_dict.get('FEATURES')
feature_input_vals={}
for i, col in enumerate(feature_input.get('Column Name').values()):
    feature_input_vals[col] = st.selectbox(col, feature_input.get('Members')[i],key=col)


input_data = dict(list(feature_input_vals.items()))

input_data= pd.DataFrame.from_dict(input_data,orient='index').T

# Prediction
if st.button('Predict'):
    prediction = predict_mushroom(input_data)[0]
    st.write(f'**{'Prediction'}**:This Mushroom is {prediction}.')