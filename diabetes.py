#install supported python libraries
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

# Load data
data=pd.read_csv('diabetes.csv')
X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values

# Train the model
classifier = KNeighborsClassifier()
classifier.fit(X, Y)

# Prediction on training data for accuracy
Y_pred = classifier.predict(X)
accuracy = accuracy_score(Y, Y_pred)

# Streamlit app
st.header('Diabetes Prediction System by Aaditya')
st.write(f"Model Accuracy: {accuracy*100:.2f}%")

# Collecting user input
pregnancies = st.text_input('Enter Pregnancies')
glucose = st.text_input('Enter Glucose')
bloodPressure = st.text_input('Enter Blood Pressure')
skinThickness = st.text_input('Enter Skin Thickness')
insulin = st.text_input('Enter Insulin')
bmi = st.text_input('Enter BMI')
dpf = st.text_input('Enter Diabetes Pedigree Function')
age = st.text_input('Enter Age')

if st.button('Predict'):
    try:
        # Convert inputs to float
        inputData = [[
            float(pregnancies),
            float(glucose),
            float(bloodPressure),
            float(skinThickness),
            float(insulin),
            float(bmi),
            float(dpf),
            float(age)
        ]]
        
        prediction = classifier.predict(inputData)[0]
        if prediction == 0:
            st.success('No Diabetes')
        else:
            st.warning('Diabetes Found')
    except ValueError as e:
        st.error(f"Invalid input: {e}")
