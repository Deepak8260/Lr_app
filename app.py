import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_model():
    with open('lr_lin_reg_model.pkl', 'rb') as file:
        model, scaler, le = pickle.load(file)
    return model, scaler, le

def preprocessing_input_data(data, scaler, le):
    data['Extracurricular Activities'] = le.transform(data['Extracurricular Activities'])
    df_transformed = scaler.transform(pd.DataFrame(data))

    return df_transformed

def predict_data(data):
    model, scaler, le = load_model()
    preprocessed_data = preprocessing_input_data(data, scaler, le)
    prediction = model.predict(preprocessed_data)

    return prediction[0]

def main():
    st.title('Student Performance Prediction By Deepak')
    st.write('Enter Your Details To Get Prediction Of Your Score')
    
    hours_studied = st.number_input('Hours Studied', min_value=1, max_value=10, value=5)
    previous_score = st.number_input('Previous Scores', min_value=40, max_value=100, value=70)
    extracurricular_activities = st.selectbox('Extracurricular Activities', ['Yes', 'No'])
    sleep_hours = st.number_input('Sleep Hours', min_value=4, max_value=10, value=7)
    question_paper = st.number_input('Question Papers', min_value=0, max_value=10, value=5)

    if st.button('Predict Your Score'):
        user_data = {
            'Hours Studied': hours_studied,
            'Previous Scores': previous_score,
            'Extracurricular Activities': [extracurricular_activities],  # Should be a list
            'Sleep Hours': sleep_hours,
            'Sample Question Papers Practiced': question_paper
        }

        user_df = pd.DataFrame(user_data)  # Convert to DataFrame
        prediction = predict_data(user_df)
        st.success(f'Your Predicted Score is: {prediction:.2f}')

if __name__ == '__main__':
    main()
