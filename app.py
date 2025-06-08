import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64

st.title("Heart Disease Predictor")
tab1,tab2=st.tabs(['predict','Model Information'])

with tab1:      
    age = st.number_input("Age (years)", min_value=0, max_value=150)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", ["Typical Angina" , "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=8, max_value=300)
    chol = st.number_input("Serum Cholesterol (mm/dl)", min_value=0)
    fbs = st.selectbox("Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"])
    restecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertropy"])
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=202)
    exang = st.selectbox("Exercise-Induced Angina", [ "No","Yes"])
    oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
    ca = st.number_input("No. of Major Vessels", min_value=0, max_value=2)
    thal = st.number_input("Thalassemia", min_value=0, max_value=3)

    # Convert categorical inputs to numeric

    sex = 0 if sex == "Male" else 1 
    cp = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(cp)
    fbs=1 if fbs == "> 120 mg/dl" else 0
    restecg = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(restecg)
    exang = 1 if exang == "Yes" else 0
    slope = ["Upsloping", "Flat", "Downsloping"].index(slope)
    
    # creating a dataframe with user input
    
    input_data1 = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak],
        'slope': [slope],
        'ca': [ca],
        'thal': [thal]
    })

    algonames = ['Logistic Regression', 'Random Forest', 'K-Nearest Neighbour', 'Support Vector Machine']
    modelnames = ['LogisticRegress.pkl', 'RandomFores.pkl','Knearestneigb.pkl', 'Supportvector.pkl']
    
    predictions = []
    def predict_heart_disease(data):
        for modelname in modelnames:
            model = pickle.load(open(modelname, 'rb'))
            prediction = model.predict(data)
            predictions.append(prediction)
        return predictions

    #Create a submit button to make predictions
    if st.button("Submit"):
        st.subheader('Results....')
        st.markdown('------------------')
        result = predict_heart_disease(input_data1)
        
        for i in range(len(predictions)):
            st.subheader(algonames[i])
            if result[i][0] == 0:
                st.write("No heart disease detected.")
            else:
                st.write("Heart disease detected.")
                st.markdown('------------------')
                
with tab2:
    import plotly.express as px
    data = {'Logistic Regression': 92.30, 'Random Forest': 87.91, 'K-Nearest Neighbour': 85.71, 'Support Vector Machine':90.10}
    Models = list(data.keys())
    Accuracies = list(data.values())
    df=pd.DataFrame(list(zip(Models,Accuracies)), columns=['Models', 'Accuracies'])
    fig = px.bar(df, x='Models', y='Accuracies')
    st.plotly_chart(fig)                