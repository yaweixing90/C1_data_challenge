import os
import pandas as pd

from six.moves import urllib


class AirbnbData:
    def __init__(self):
        self.__housing_url = "http://data.insideairbnb.com/united-states/ny/new-york-city/2019-07-08/data/listings.csv.gz"
        self.housing_path = os.path.join("datasets", "airbnb")
        self.data = pd.DataFrame()

    def fetchAirbnbData(self):
        if ~self.checkExistAirbnbData():
            os.makedirs(self.housing_path, exist_ok=True)
            gz_path = os.path.join(self.housing_path, "listings.csv.gz")
            urllib.request.urlretrieve(self.__housing_url, gz_path)
        else:
            print("Airbnb Data has been downloaded before. No need to fetch again.")

    def checkExistAirbnbData(self):
        if os.path.exists(os.path.join(self.housing_path, "listings.csv.gz")):
            return True
        else:
            return False

    def loadAirbnbData(self):
        csv_path = os.path.join(self.housing_path, "listings.csv.gz")
        self.data = pd.read_csv(csv_path, compression='gzip', error_bad_lines=False, low_memory=False)

