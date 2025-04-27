import streamlit as st
import pickle
import os

# Load the trained model
model = pickle.load(open(os.path.join('model', 'salary_predictor.pkl'), 'rb'))

st.title('Salary Predictor App')

# User inputs
exp = st.number_input('Experience (in years)', min_value=0, step=1)
test_score = st.number_input('Test Score (out of 10)', min_value=0, max_value=10, step=1)
interview_score = st.number_input('Interview Score (out of 10)', min_value=0, max_value=10, step=1)

# Predict button
if st.button('Predict Salary'):
    prediction = model.predict([[exp, test_score, interview_score]])
    st.success(f'Predicted Salary: ${int(prediction[0])}')
