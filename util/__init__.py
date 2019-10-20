import pandas as pd
import plotly.graph_objects as go
from pandas import DataFrame
import plotly.express as px

def missingAnalysis(df:DataFrame):
    missing = df.isnull().sum().to_frame('missing_count')
#     missing.reset_index(level=0, inplace=True)
    missing['missing_perc'] = missing['missing_count']/df.shape[0] * 100
    r = missing.sort_values(by='missing_perc', ascending=False)
    fig = go.Figure(go.Bar(x=r.index, y = r.missing_perc))
    fig.update_layout(
        title='MISSING SUMMARY',
        xaxis_tickfont_size=14,
        xaxis=dict(
            title="Variable Name",
        ),
        yaxis=dict(
            title='Missing Percentage',
            titlefont_size=16,
            tickfont_size=14,
            ticksuffix="%"
        ),
        legend=dict(
            x=0,
            y=1.0,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
        )
    )     
    fig.show()
    return r

def basicInfoAnalysis(df:DataFrame):
    numeric_col_num = len(df._get_numeric_data().columns)
    object_col_num = len(df.select_dtypes(['object']).columns)
    categorical_col_num = len(df.select_dtypes(['category']).columns)
    bool_col_num = len(df.select_dtypes(['bool']).columns)
    print('# of numeric columns:', numeric_col_num)
    print('# of object columns:', object_col_num)
    print('# of category columns:', categorical_col_num)
    print('# of bool columns:', bool_col_num)
    
    if numeric_col_num != 0:
        print('*'*10+' Numeric Variable Insight '+'*'*10)
        print(df[df._get_numeric_data().columns].describe())
    
    # if object_col_num != 0:
    #     print('*'*10+' Object Variable Insight '+'*'*10)
    #     for col in df.select_dtypes(['object']).columns:
    #         print('*'*10 + str(col) + '*'*10)
    #         print(df[col].value_counts(normalize=True, dropna=False))
    #
    # if categorical_col_num != 0:
    #     print('*'*10+' Categorical Variable Insight '+'*'*10)
    #     for col in df.select_dtypes(['category']).columns:
    #         print('*'*10 + str(col) + '*'*10)
    #         print(df[col].value_counts(normalize=True, dropna=False))
    #
    # if bool_col_num != 0:
    #     print('*'*10+' Boolean Variable Insight '+'*'*10)
    #     for col in df.select_dtypes(['category']).columns:
    #         print('*'*10 + str(col) + '*'*10)
    #         print(df[col].value_counts(normalize=True, dropna=False))
            
            
def categorical_count_plot(df:DataFrame, column:str):
    counts = df[column].value_counts(normalize=True, dropna=False).reset_index()
    fig = go.Figure(go.Bar(x=counts['index'], y = counts[column]))
    fig.update_layout(
        title=column + ' summary',
        xaxis_tickfont_size=14,
        xaxis=dict(
            title = column,
            type='category'
        ),
        yaxis=dict(
            title='percentage',
            titlefont_size=16,
            tickfont_size=14,
            ticksuffix="%"
        ),
        legend=dict(
            x=0,
            y=1.0,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
        )
    )     
    fig.show()

def nyc_map_plot(df, color=None, size=None): 
    px.set_mapbox_access_token(
    "pk.eyJ1IjoieWF3ZWl4aW5nIiwiYSI6ImNrMW8zcG1wejBpN3czbXEzZXlxNDB5eWUifQ.mtATdzxTQANOPjTMrVdQBA")
    fig = px.scatter_mapbox(df,
                        lat="latitude",
                        lon="longitude",
                        color=color,
                        color_continuous_scale='Magma', 
                        size=size,
                        size_max=15,
                        zoom=10)
    fig.show()
    
    
    
