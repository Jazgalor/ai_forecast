import pandas as pd
import numpy as np
import random as rn
import time
from ast import literal_eval
from modules.algorithms import Clusterization
from modules.dataProcessing import DataProcessing
from modules.fetcher import Fetcher
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--n-days",
                    help="predict next n days given. if not specified 4 days forward is default value")
parser.add_argument("-k", "--n-neighbours",
                    help="predict next n days given. if not specified 48 neighbours is default value")
parser.add_argument("-d", "--data",
                    help="get real time data", action="store_true")
args = parser.parse_args()

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
def predict_n_times(_range,_data,_base,_n, k):
    l = []
    for i in tqdm(range(_n)):
        DataProcessing.normalize_element(_base,_range)
        _base = Clusterization.knn_reglin_forecast(_data,_base,k)
        DataProcessing.denormalize_element(_base,_range)
        # print(f"progress {'#'*(i+1)}{'.'*(_n-i-1)}")
        l.append(tuple(_base))  
    weather_table = pd.DataFrame(l, columns=["temp", "humidity", "precip", "sealevelpressure", "windspeed", "cloudcover"])
    print(weather_table)

if args.data:
    print("fetching real time data to overwrite:")
    test = Fetcher.fetch_data()
    test.to_csv("./data/daily.csv", sep=',', encoding='utf-8', index=False)
    print("fetched data without any error")

# processing + get min max from set (to revert normalization)
rangetab=first_time_process("./data/daily")

# convert string dataset as python literal
data= pd.read_csv("./data/daily_fitted.csv", sep=';')
for col in data.columns:
    data[col] = data[col].apply(lambda x: literal_eval(str(x)))

# place for input data
test=[12,30,0,1000,5,80]
k = 48
if args.n_neighbours:
    k = args.n_neighbours
if args.n_days:
    predict_n_times(rangetab,data,test,int(args.n_days), int(k))
else:
    predict_n_times(rangetab,data,test,4, int(k))

# for test and way of testing check 'test.py'
#EOF