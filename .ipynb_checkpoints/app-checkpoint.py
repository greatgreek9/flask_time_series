# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 15:16:10 2021

@author: sunbeam
"""

import pandas as pd
from flask import Flask, jsonify, request, render_template
import pickle

import plotly
import plotly.graph_objs as go

import numpy as np
import json


# load model
#model = pickle.load(open('my_time_series_model.h5', 'rb'))


# app
app = Flask(__name__, template_folder='templates')




@app.route("/", methods=["GET"])
def get_register():
    return render_template('index1.html')

@app.route("/second", methods=["GET"])
def get_second():
    return render_template('second.html')


if __name__ == '__main__':
    app.run(port = 5000, debug=True)