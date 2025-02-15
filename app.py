import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://user:password_password@cluster0.7tqwf.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri, server_api=ServerApi('1'))
db=client['student']
collection = db['student_pred']

def load_model():
    with open('lr_lin_reg_model.pkl','rb') as file:
        model, scaler, le = pickle.load(file)
    return model, scaler, le

def preprocessing_input_data(data,scaler,le):
    df=pd.DataFrame([data])
    df['Extracurricular Activities'] = le.transform(df['Extracurricular Activities'])[0]
    df_transformed = scaler.transform(df)

    return df_transformed

def predict_data(data):
    model, scaler, le = load_model()
    preprocessed_data = preprocessing_input_data(data,scaler,le)
    prediction = model.predict(preprocessed_data)

    return prediction[0]


def main():
    st.title('Student Performance Prediction')
    st.write('Enter Your Details To Get Prediction Of Your Input')
    
    hours_studied = st.number_input('Hours Studied',min_value=1,max_value=10,value=5)
    Previosu_score = st.number_input('Previous Scores',min_value=40,max_value=100,value=70)
    Extracurricular_Activities = st.selectbox('Extracurricular Activities',['Yes','No'])
    Sleep_Hours = st.number_input('Sleep Hours',min_value=4,max_value=10,value=7)
    Question_paper = st.number_input('Question Papers',min_value=0,max_value=10,value=5)


    if st.button('Predict_your_score'):
        user_data={
            'Hours Studied' : hours_studied,
            'Previous Scores' : Previosu_score,
            'Extracurricular Activities': Extracurricular_Activities,
            'Sleep Hours' : Sleep_Hours,
            'Sample Question Papers Practiced' : Question_paper

        }
        prediction = predict_data(user_data)
        user_data['prediction'] = prediction
        collection.insert_one(user_data)
        
        st.success(f'Your Prediction is {prediction:.2f}')
        



if __name__ == '__main__':
    main()

