import pandas as pd
import numpy as np
import random as rn
import time
from ast import literal_eval
from modules.algorithms import Clusterization
from modules.dataProcessing import DataProcessing

#function to create sub datasets no shorten the time
def first_time_process(csv_name):
    data = pd.read_csv(csv_name+".csv")
    data_reduced = data[["temp","humidity","precip","sealevelpressure","windspeed","cloudcover"]]
    DataProcessing.normalization_to_1(data_reduced)
    data_fit = DataProcessing.fit(data_reduced)
    data_fit.to_csv(csv_name+"_fitted.csv", sep=';', index=False)


#convert string dataset as python literal
data= pd.read_csv("./data/hourly_fitted.csv", sep=';')
for col in data.columns:
    data[col] = data[col].apply(lambda x: literal_eval(str(x)))

#shuffle data
DataProcessing.shuffle(data)
trainingData, testingData = DataProcessing.split(data, 0.7)


#testing area
test = [2,4,6,8]
for x in test:
    _start = time.perf_counter()
    allAvgPrec = 0
    for i in range(len(testingData)):
        avgPrec = 0
        result = Clusterization.knn_forecast(trainingData.copy(), testingData.iloc[i], x)  
        for j in range(len(result)):
            avgPrec += (1 - (np.abs(testingData.iloc[i][j][1] - result[j])/testingData.iloc[i][j][1]))
        allAvgPrec += avgPrec/len(result)
    print(allAvgPrec/len(testingData),time.perf_counter()-_start)