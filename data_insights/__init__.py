from data_init.airbnb_data import AirbnbData
from data_init.zillow_data import ZillowData
import pandas as pd

zillow = ZillowData()
zillow.loadZillowData()

airbnb = AirbnbData()
airbnb.fetchAirbnbData()
airbnb.loadAirbnbData()

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_columns', 500)

zillowInfo = zillow.data.describe().reset_index()
airbnbInfo = airbnb.data.describe().reset_index()



