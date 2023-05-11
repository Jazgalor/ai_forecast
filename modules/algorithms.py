import pandas as pd

class Clusterization:
    @staticmethod
    def knn_forecast(x, sample, k):
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