# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 23:23:41 2023

@author: starj
"""

import numpy as np
import pickle
import streamlit

loaded_model=pickle.load(open('E:/Diabet_Strmlit/trained_model.sav','rb'))

def diabetes_prediction(input_data):

        input_data_as_numpy_array = np.asarray(input_data)

        input_data_reshape = input_data_as_numpy_array.reshape(1,-1)

        prediction= loaded_model.predict(input_data_reshape)
        print(prediction)

        if(prediction[0]==0):
          return("Non Diabetic")
        else:
          return('Diabetic')
        