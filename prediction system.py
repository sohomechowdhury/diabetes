# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pickle
import numpy as np


loaded_model=pickle.load(open('E:/Diabet_Strmlit/trained_model.sav','rb'))
scaler = pickle.load(open('E:/Diabet_Strmlit/scaler.pkl', 'rb'))

input_data= (1,103,30,38,83,43.3,0.183,33)

input_data_as_numpy_array = np.asarray(input_data)

input_data_reshape = input_data_as_numpy_array.reshape(1,-1)
std_data = scaler.transform(input_data_reshape)
prediction= loaded_model.predict(std_data)
print(prediction)

if(prediction[0]==0):
  print("Non Diabetic")
else:
  print('Diabetic')