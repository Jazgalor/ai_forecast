import pandas as pd
import random as rn

class DataProcessing:
    @staticmethod
    def shuffle(x):
        i = len(x)
        while i>0:     
            r = rn.randint(0,i-1)
            x.iloc[i-1], x.iloc[r] = x.iloc[r], x.iloc[i-1]
            i-=1
            
    @staticmethod
    def split(x,k):
        return x[:int(len(x)*k)], x[int(len(x)*k):]
    
    @staticmethod
    def normalization(x):
        values = x.iloc[:,:]
        columnNames = values.columns.tolist()
        for column in columnNames:
            data = values.loc[:,column]
            datMin = min(data)
            datMax = max(data)
            for row in range(0,len(data),1):
                x.at[row,column] = ((x.at[row,column]-datMin)/(datMax-datMin))

    @staticmethod
    def normalization_to_1(x):
        values = x.iloc[:,:]
        columnNames = values.columns.tolist()
        for column in columnNames:
            data = values.loc[:,column]
            datMin = min(data)
            datMax = max(data)
            for row in range(0,len(data),1):
                x.at[row,column] = ((x.at[row,column]-datMin)/(datMax-datMin)) + 1

    @staticmethod
    def fit(x):
        new=[]
        for col in x.columns:
            newr = []
            for i in range(len(x)-1):
                newr.append((x.iloc[i][col], x.iloc[i+1][col]))
            new.append(newr)
        result = pd.DataFrame(data={x.columns[i]: new[i] for i in range(len(new))})
        return result
    
