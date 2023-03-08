# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 23:23:41 2023

@author: starj
"""

import numpy as np
import pickle
import streamlit as st

loaded_model=pickle.load(open('E:/Diabet_Strmlit/trained_model.sav','rb'))
scaler = pickle.load(open('E:/Diabet_Strmlit/scaler.pkl', 'rb'))

def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshape = input_data_as_numpy_array.reshape(1,-1)
       
    std_data = scaler.transform(input_data_reshape)
       
    prediction= loaded_model.predict(std_data)
       
    print(prediction)
    
    if(prediction[0]==0):
        return("Non Diabetic")
    else:
         return('Diabetic')
      
        
      
def main():
    st.title('Diabetes prediction Web App')
    
    Pregnancies = st.text_input("Number of Pregnancies") 
    Glucose = st.text_input("Glocose value")
    BloodPressure = st.text_input("BP value")
    SkinThickness = st.text_input("Skin thickness value")
    Insulin = st.text_input("Insulin value")
    BMI = st.text_input("Bmi Value")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function Value")
    Age= st.text_input("Age")
    
    diagnosis = ''
    
    if(st.button('Test Result')):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
    st.success(diagnosis)
    
    
if __name__=='__main__':
    main()