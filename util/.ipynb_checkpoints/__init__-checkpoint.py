import pandas as pd
import plotly.express as px
from pandas import DataFrame


def missingAnalysis(df:DataFrame):
    missing = df.isnull().sum().to_frame('missing_count')
    missing.sort_values(by='missing_count')
    missing.reset_index(level=0, inplace=True)
    fig = px.bar(missing, x='index', y = 'missing_count')
    fig.show()
    return missing

