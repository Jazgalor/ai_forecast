import pandas as pd
import numpy as np

class Clusterization:
    @staticmethod
    def knn_add_forecast(x, sample, k):
        distances = []
        for i in range(len(x)):
            tmp = 0
            for j in range(len(sample) - 1):
                tmp += (x.iloc[i][j][0] - sample[j][0]) ** 2
            distances.append(tmp ** (1 / 2))
        x['distances'] = distances
        nearest = x.sort_values(by="distances").iloc[:k]
        # print(nearest)
        forecast = []
        for col in nearest.columns[:-1]:
            test = 0
            for i in range(k):
                test += nearest.iloc[i][col][1] - nearest.iloc[i][col][0]
            forecast.append(test/k)
        return [sample[i][0]+forecast[i] for i in range(len(forecast))]

    @staticmethod
    def knn_mul_forecast(x, sample, k):
        distances = []
        for i in range(len(x)):
            tmp = 0
            for j in range(len(sample) - 1):
                tmp += (x.iloc[i][j][0] - sample[j][0]) ** 2
            distances.append(tmp ** (1 / 2))
        x['distances'] = distances
        nearest = x.sort_values(by="distances").iloc[:k]
        # print(nearest)
        forecast = []
        for col in nearest.columns[:-1]:
            test = 0
            for i in range(k):
                test += nearest.iloc[i][col][1]/nearest.iloc[i][col][0]
            forecast.append(test/k)
        return [sample[i][0]*forecast[i] for i in range(len(forecast))]
    
    @staticmethod
    def knn_reglin_forecast(x, sample, k):
        distances = []
        for i in range(len(x)):
            tmp = 0
            for j in range(len(sample) - 1):
                tmp += (x.iloc[i][j][0] - sample[j][0]) ** 2
            distances.append(tmp ** (1 / 2))
        x['distances'] = distances
        nearest = x.sort_values(by="distances").iloc[:k]
        #print(nearest)
        forecast = []
        for col in nearest.columns[:-1]:
            transformed = np.array(nearest[col].tolist()).transpose()
            forecast.append(Regression.linear_ls(transformed[0],transformed[1]))
        return [sample[i][0]*forecast[i][0]+forecast[i][1] for i in range(len(forecast))]


class Regression:
    @staticmethod
    def linear_ls(_x,_y):
        s = len(_y)
        sx = np.sum(_x)
        sy = np.sum(_y)
        sxy = np.sum([_x[i]*_y[i] for i in range(len(_y))])
        sxx = np.sum([x**2 for x in _x])
        d = s*sxx-sx**2
        if d == 0: return (1,0)
        a = (s*sxy-sx*sy)/d
        b = (sxx*sy-sx*sxy)/d
        return (a,b)

