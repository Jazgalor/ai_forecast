#This file should be copy of main.py (mainly for testing purposes and getting numerical data from tests)

import pandas as pd
import numpy as np
import random as rn
import time
from ast import literal_eval
from modules.algorithms import Clusterization
from modules.dataProcessing import DataProcessing

# function to create sub datasets no shorten the time
def first_time_process(csv_name):
    data = pd.read_csv(csv_name+".csv")
    data_reduced = data[["temp","humidity","precip","sealevelpressure","windspeed","cloudcover"]]
    DataProcessing.normalization_to_1(data_reduced)
    data_fit = DataProcessing.fit(data_reduced)
    data_fit.to_csv(csv_name+"_fitted.csv", sep=';', index=False)


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

