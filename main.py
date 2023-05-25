import pandas as pd
import numpy as np
import random as rn
import time
from ast import literal_eval
from modules.algorithms import Clusterization
from modules.dataProcessing import DataProcessing

# function to create sub datasets no shorten the time RETURNS list of min/max
def first_time_process(csv_name):
    data = pd.read_csv(csv_name+".csv")
    data_reduced = data[["temp","humidity","precip","sealevelpressure","windspeed","cloudcover"]]
    rangetab = DataProcessing.get_range(data_reduced)
    DataProcessing.normalization_to_1(data_reduced)
    data_fit = DataProcessing.fit(data_reduced)
    data_fit.to_csv(csv_name+"_fitted.csv", sep=';', index=False)
    return rangetab

#function to bulk predict n times
def predict_n_times(_range,_data,_base,_n):
    #base is a list of 6 parameters of dataset, but it is not a list of tuples
    k=0
    #choose daily/hourly dataset option
    x = input()
    if x == "0": k=48
    else: k=7
    print(k)
    for i in range(_n):
        DataProcessing.normalize_element(_base,_range)
        _base = Clusterization.knn_reglin_forecast(_data,_base,k)
        DataProcessing.denormalize_element(_base,_range)
        print(f"Prediction #{i+1}: {_base}")

# processing + get min max from set (to revert normalization)
rangetab=first_time_process("./data/hourly")

# convert string dataset as python literal
data= pd.read_csv("./data/hourly_fitted.csv", sep=';')
for col in data.columns:
    data[col] = data[col].apply(lambda x: literal_eval(str(x)))

# place for input data
test=[12,30,0,1000,5,80]
predict_n_times(rangetab,data,test,4)

# for test and way of testing check 'test.py'
#EOF