#This file should be copy of main.py (mainly for testing purposes and getting numerical data from tests)

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


rangetab=first_time_process("./data/hourly")

# convert string dataset as python literal
data= pd.read_csv("./data/hourly_reduced_fitted.csv", sep=';')
for col in data.columns:
    data[col] = data[col].apply(lambda x: literal_eval(str(x)))

# shuffle data
DataProcessing.shuffle(data)
trainingData, testingData = DataProcessing.split(data, 0.7)


# testing area
k_range = [1,2,int(len(testingData)/4),int(len(testingData)/2),int(len(testingData))]
for x in k_range:
    test = [0,0,0,0]
    for n in range(10):
        DataProcessing.shuffle(data)
        trainingData, testingData = DataProcessing.split(data, 0.7)
        _start = time.perf_counter()
        allAvgPrec = 0
        for i in range(len(testingData)):
            avgPrec = 0
            result = Clusterization.knn_add_forecast(trainingData.copy(), testingData.iloc[i], x) 
            for j in range(len(result)):
                avgPrec += (1 - (np.abs(testingData.iloc[i][j][1] - result[j])/testingData.iloc[i][j][1]))
            allAvgPrec += avgPrec/len(result)
        test[2] += allAvgPrec/len(testingData)
        test[3] += time.perf_counter()-_start 
        _start = time.perf_counter()
        allAvgPrec = 0
        for i in range(len(testingData)):
            avgPrec = 0
            result = Clusterization.knn_reglin_forecast(trainingData.copy(), testingData.iloc[i], x) 
            for j in range(len(result)):
                avgPrec += (1 - (np.abs(testingData.iloc[i][j][1] - result[j])/testingData.iloc[i][j][1]))
            allAvgPrec += avgPrec/len(result)
        test[0] += allAvgPrec/len(testingData)
        test[1] += time.perf_counter()-_start 
    print(Clusterization.knn_add_forecast.__name__,x,test[2]/10,test[3]/10)
    print(Clusterization.knn_reglin_forecast.__name__,x,test[0]/10,test[1]/10)
