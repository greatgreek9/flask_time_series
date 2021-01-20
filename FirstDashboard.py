from flask import Flask, render_template,request
import plotly
import plotly.graph_objs as go

import pandas as pd
import numpy as np
import json

from datetime import datetime
import tensorflow as tf
from tensorflow import keras

df = pd.read_csv("/home/sunbeam/Documents/STOCK_DATASETS_FINAL/AAPL.csv")
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

print("intermediate_predictions")
print(scaler.inverse_transform(predictions))


app = Flask(__name__)

@app.route('/tsla')
def get_tsla():
    feature = 'Bar'
    bar = create_plot(feature)
    return render_template('tsla.html', plot=bar)

@app.route('/nflx')
def get_nflx():
    feature = 'Bar'
    bar = create_plot(feature)
    return render_template('nflx.html', plot=bar)

@app.route('/amrn')
def get_amrn():
    feature = 'Bar'
    bar = create_plot(feature)
    return render_template('amrn.html', plot=bar)



@app.route('/twitter')
def apple_analytics():
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
        
    print("LST PRINTED")
    print(lst_output)
    final_y = scaler.inverse_transform(lst_output).reshape(1,-1).tolist()[0]

    feature = 'Bar'
    scatter = create_plot(feature,pre_date,final_y)
    return render_template('twitter.html', plot=scatter)


    
@app.route('/')
def index():
    feature = 'Bar'
    bar = create_plot(feature)
    return render_template('index.html', plot=bar)

def create_plot(feature,x=x,y=y):
    if feature == 'Bar':
        data = [
            go.Scatter(x=x, y=y, mode='markers')
        ]
    else:
        N = 1000
        random_x = np.random.randn(N)
        random_y = np.random.randn(N)

        # Create a trace
        data = [go.Scatter(
            x = random_x,
            y = random_y,
            mode = 'markers'
        )]


    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

@app.route('/bar', methods=['GET', 'POST'])
def change_features():

    feature = request.args['selected']
    graphJSON= create_plot(feature)
    return graphJSON

if __name__ == '__main__':
    app.run(port = 5000, debug=True)
