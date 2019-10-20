import pandas as pd

class ZillowData:
    def __init__(self):
        self.__local_file_path = 'datasets/zillow/Zip_Zhvi_2bedroom.csv'
        self.data = pd.DataFrame()

    def loadZillowData(self):
        self.data = pd.read_csv(self.__local_file_path)


