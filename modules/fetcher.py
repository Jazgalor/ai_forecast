import pandas as pd
import random as rn
import requests
from io import StringIO
import datetime

class Fetcher:
    @staticmethod
    def fetch_data():
        today = datetime.datetime.today().strftime('%Y-%m-%d')
        start_date = (datetime.datetime.now() - datetime.timedelta(300)).strftime('%Y-%m-%d')
        print(start_date)
        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/Gliwice/{start_date}/{today}?unitGroup=us&include=days&key=C2VGKJ6PKP32QTEK65NPPXVN8&contentType=csv"
        res = requests.get(url)
        try:
            response = pd.read_csv(StringIO(res.text))
        #     response = response[["temp", "humidity", "precip",
        #                         "sealevelpressure", "windspeed", "cloudcover"]]
            if len(response) == 0:
                raise Exception(response.loc[0])
            return response
        except:
            raise Exception(res.text)