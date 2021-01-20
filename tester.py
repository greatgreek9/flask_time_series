#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 19:10:43 2021

@author: sunbeam
"""

import plotly.graph_objects as go

import plotly.io as pio
#pio.renderers.default = 'svg'
pio.renderers.default = 'browser'
import pandas as pd

import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime

df = pd.read_csv("/home/sunbeam/Documents/STOCK_DATASETS_FINAL/AAPL.csv",parse_dates=[0])

print(df)
pre_date =pd.Series(pd.date_range(datetime.today(), periods=30))
x = df['Date']
y = df['Close']

pre_date.shape

x.shape
y.shape

input_data_date = df.Date[len(df)-100:]
input = df.Close

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(input).reshape(-1,1))

#input_data = np.array(df1[len(df1)-100-1:len(df1)-1]).reshape(-1,1)
input_data = np.array(df1[len(df1)-100:]).reshape(-1,1)
print(df1[-1])
print(input_data)
print(len(input_data))


model = keras.models.load_model('my_time_series_model.h5')

predictions = model.predict(np.array(input_data).reshape(1,-1).reshape((1,100,1)))

print(scaler.inverse_transform(predictions))

x_input = np.array(input_data[:]).reshape(1,-1)
x_input.shape

temp_input = list(x_input)
temp_input=temp_input[0].tolist()


# demonstrate prediction for next 10 days
from numpy import array

lst_output=[]
n_steps=100
i=0
while(i<30):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    

print(lst_output)
final_y = scaler.inverse_transform(lst_output).reshape(1,-1).tolist()[0]

fig = go.Figure(data=go.Scatter(x=pre_date, y=final_y, mode='markers'))

fig.show()