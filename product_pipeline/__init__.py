import re

from data_init.airbnb_data import AirbnbData
from data_init.zillow_data import ZillowData
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import datetime

zillow = ZillowData()
zillow.loadZillowData()

airbnb = AirbnbData()
airbnb.fetchAirbnbData()
airbnb.loadAirbnbData()

airbnb_filter = airbnb.data['bedrooms'] == 2.0
airbnb_data = airbnb.data[airbnb_filter].copy()

zillow_filter = zillow.data['City'] == 'New York'
zillow_data = zillow.data[zillow_filter].copy()

selected_feature = [
    "neighbourhood_group_cleansed", "zipcode", "latitude",
    "longitude", "price", "weekly_price", "monthly_price", "availability_30",
    "availability_60", "availability_90", "availability_365", "square_feet"
]

neighbourhood_group_cleansed_ix, zipcode_ix, latitude_ix, \
longitude_ix, price_ix, weekly_price_ix, monthly_price_ix, \
availability_30_ix, availability_60_ix, availability_90_ix, \
availability_365_ix, square_feet_ix = [
    list(airbnb_data.columns).index(col) for col in selected_feature
]


class FeatureCleaning(BaseEstimator, TransformerMixin):
    def __init__(self, price_ix, weekly_price_ix, monthly_price_ix,
                 zipcode_ix):
        self.price_ix = price_ix
        self.weekly_price_ix = weekly_price_ix
        self.monthly_price_ix = monthly_price_ix
        self.zipcode_ix = zipcode_ix

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for ix in [self.price_ix, self.weekly_price_ix, self.monthly_price_ix]:
            X.iloc[:, ix] = X.iloc[:, ix].str.replace("$", "").str.replace(
                ",", "").astype(float)

        X.iloc[:, zipcode_ix] = X.iloc[:, zipcode_ix].str[:5]
        X = X.loc[~X.iloc[:, zipcode_ix].isnull(), :]
        return X


def limit_bedrooms(X, n=2):
    return X[X['bedrooms'] == n]


def get_val_prices_ix(X, price_ix, neighbourhood_group_cleansed_ix=neighbourhood_group_cleansed_ix, lb=.1, ub=.9):
    no_zero = X.iloc[:, price_ix] != 0
    val_ix = no_zero
    for v in X.iloc[:, neighbourhood_group_cleansed_ix].unique():
        ngc_ix = X.iloc[:, neighbourhood_group_cleansed_ix] == v
        val_ix[ngc_ix] = (no_zero[ngc_ix]) & (X.loc[no_zero & ngc_ix, X.columns[price_ix]].between(
            X.loc[no_zero & ngc_ix, X.columns[price_ix]].quantile(lb),
            X.loc[no_zero & ngc_ix, X.columns[price_ix]].quantile(ub)))

    return val_ix


class SquareFeetImputation(BaseEstimator, TransformerMixin):
    def __init__(self, lb, ub):
        self.lb = lb
        self.ub = ub

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        # Find index of the outliers regarding the price variables
        val_price_ix = get_val_prices_ix(X, price_ix,
                                         neighbourhood_group_cleansed_ix,
                                         self.lb, self.ub)
        val_weekly_price_ix = get_val_prices_ix(
            X, weekly_price_ix, neighbourhood_group_cleansed_ix, self.lb,
            self.ub)
        val_monthly_price_ix = get_val_prices_ix(
            X, monthly_price_ix, neighbourhood_group_cleansed_ix, self.lb,
            self.ub)
        # mark price variables are outlier or not.
        # use integer to store multiple boolean columns
        X['collapses_price_val'] = val_price_ix.values.astype(int) + (
                val_weekly_price_ix.values.astype(int) << 1) + (
                                           val_monthly_price_ix.values.astype(int) << 2)

        val_col_row_pair = {
            price_ix: val_price_ix,
            weekly_price_ix: val_weekly_price_ix,
            monthly_price_ix: val_monthly_price_ix
        }

        for col_ix in val_col_row_pair.keys():
            val_row = val_col_row_pair[col_ix]
            # among validated price, weekly price, monthly_price rows, get no missing square_feet data
            sqrf_no_missing = X[(~X.iloc[:, square_feet_ix].isnull())
                                & (val_row) & (X.iloc[:, square_feet_ix] != 0)]
            """
            group by neighbourhood_group_cleansed, sum square feet and one of the price variables 
            where this selected price is qualified
            """
            agg_df = sqrf_no_missing.groupby(
                sqrf_no_missing.columns[neighbourhood_group_cleansed_ix]).agg({
                sqrf_no_missing.columns[square_feet_ix]:
                    'sum',
                sqrf_no_missing.columns[col_ix]:
                    'sum'
            }).reset_index()

            agg_df['price_per_sqrf'] = agg_df[
                                           sqrf_no_missing.columns[col_ix]] / agg_df['square_feet']

            sqrf_missing_ix = (X.iloc[:, square_feet_ix].isnull()) | (X.iloc[:, square_feet_ix] == 0)

            """for each neighborhood using average price_per_sqrt is constant assumption
            calculating square feet with price > weekly_price > monthly_price"""

            for neighborhood in agg_df.neighbourhood_group_cleansed:
                X.loc[(X.neighbourhood_group_cleansed == neighborhood)
                      & (sqrf_missing_ix) &
                      (val_row), X.columns[square_feet_ix]] = X.loc[
                                                                  (X.neighbourhood_group_cleansed == neighborhood)
                                                                  & (sqrf_missing_ix)
                                                                  & (val_row), X.columns[col_ix]] / agg_df.loc[
                                                                  agg_df.neighbourhood_group_cleansed ==
                                                                  neighborhood, 'price_per_sqrf'].values

        return X[
            ~X.iloc[:, square_feet_ix].isnull()]  # drop square feet missing according to previous plots


feature_pipeline = Pipeline([('feature_clean',
                              FeatureCleaning(price_ix, weekly_price_ix,
                                              monthly_price_ix, zipcode_ix)),
                             ('limit_bedrooms',
                              FunctionTransformer(limit_bedrooms,
                                                  validate=False,
                                                  kw_args={'n': 2})),
                             ('sqrf_impute', SquareFeetImputation(lb=.1, ub=.9))
                             ])

airbnb_prepared_final = feature_pipeline.fit_transform(airbnb.data.copy())


def get_val_avail_ix(X, avail_ix, lb=0.1, ub=0.9):
    no_missing_or_zero_ix = (X.iloc[:, avail_ix] != 0) & (~X.iloc[:, avail_ix].isnull())
    return no_missing_or_zero_ix & (
        X.iloc[:, avail_ix].between(X.loc[no_missing_or_zero_ix, X.columns[avail_ix]].quantile(lb),
                                    X.loc[no_missing_or_zero_ix, X.columns[avail_ix]].quantile(ub)))


class CalAvail365(BaseEstimator, TransformerMixin):
    def __init__(self, lb, ub):
        self.lb = lb
        self.ub = ub

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # get qualified availability index
        val_avail_30 = get_val_avail_ix(X, availability_30_ix, self.lb,
                                        self.ub)
        val_avail_60 = get_val_avail_ix(X, availability_60_ix, self.lb,
                                        self.ub)
        val_avail_90 = get_val_avail_ix(X, availability_90_ix, self.lb,
                                        self.ub)
        val_avail_365 = get_val_avail_ix(X, availability_365_ix, self.lb,
                                         self.ub)
        # availability_365 not qualified, but availability_90 qualified
        X.loc[(~val_avail_365) &
              (val_avail_90), X.columns[availability_365_ix]] = X.loc[
                                                                    (~val_avail_365) &
                                                                    (val_avail_90), X.columns[
                                                                        availability_90_ix]] / 90 * 365
        # availability_365 and availability_90 not qualified, but availability_60 qualified
        X.loc[(~val_avail_365) & (~val_avail_90) &
              (val_avail_60), X.columns[availability_365_ix]] = X.loc[
                                                                    (~val_avail_365) & (~val_avail_90) &
                                                                    (val_avail_60), X.columns[
                                                                        availability_60_ix]] / 60 * 365
        # availability_365, availability_90 and availability_60 not qualified, but availability_30 qualified
        X.loc[(~val_avail_365) & (~val_avail_90) & (~val_avail_60) &
              (val_avail_30), X.columns[availability_365_ix]] = X.loc[
                                                                    (~val_avail_365) & (~val_avail_90) & (
                                                                        ~val_avail_60) &
                                                                    (val_avail_30), X.columns[
                                                                        availability_30_ix]] / 30 * 365
        return X[get_val_avail_ix(X, availability_365_ix, self.lb, self.ub)]


def yearly_revenue_generate(X, customer_ratio, occupancy_rate):
    # build customer weights matrix
    customer_weights_matrix = np.array(
        [[0, 0, 0], [0, 0, 1], [0, 1, 0],
         [
             0, customer_ratio['week'] /
                (customer_ratio['week'] + customer_ratio['day']),
                customer_ratio['day'] /
                (customer_ratio['week'] + customer_ratio['day'])
         ], [1, 0, 0],
         [
             customer_ratio['month'] /
             (customer_ratio['month'] + customer_ratio['day']), 0,
             customer_ratio['day'] /
             (customer_ratio['month'] + customer_ratio['day'])
         ],
         [
             customer_ratio['month'] /
             (customer_ratio['month'] + customer_ratio['week']),
             customer_ratio['week'] /
             (customer_ratio['month'] + customer_ratio['week']), 0
         ],
         [
             customer_ratio['month'] /
             (customer_ratio['month'] + customer_ratio['week'] +
              customer_ratio['day']), customer_ratio['week'] /
             (customer_ratio['month'] + customer_ratio['week'] +
              customer_ratio['day']), customer_ratio['day'] /
             (customer_ratio['month'] + customer_ratio['week'] +
              customer_ratio['day'])
         ]])
    # using np.dot calculate dot product between price matrix and weights matrix
    # then multiple availability and occupancy
    for i in X['collapses_price_val'].drop_duplicates():
        rix = X['collapses_price_val'] == i
        cix = [monthly_price_ix, weekly_price_ix, price_ix]
        X.loc[rix, 'est_yearly_revenue'] = np.dot(
            np.nan_to_num(X.loc[rix, X.columns[cix]].values),
            customer_weights_matrix[i, :]
        ) * X.loc[rix, X.columns[availability_365_ix]].values * occupancy_rate

    return X[X['collapses_price_val'] != 0]


revenue_pipeline = Pipeline([('avail_cal', CalAvail365(0.1, 0.9)),
                             ('gen_revenue',
                              FunctionTransformer(yearly_revenue_generate,
                                                  validate=False,
                                                  kw_args={
                                                      'customer_ratio': {
                                                          'day': 3,
                                                          'week': 1,
                                                          'month': 1
                                                      },
                                                      'occupancy_rate': 0.75
                                                  }))])
feature_revenue_pipeline = make_pipeline(feature_pipeline, revenue_pipeline)
airbnb_revenue = revenue_pipeline.fit_transform(airbnb_prepared_final.copy())


def current_median(X, window_start):
    time_col = [x for x in X.columns if re.search('\d{4}-\d{2}', x)]
    X = X[["RegionName"] + time_col]
    X.index = X.RegionName
    zillow_data_cleaned = X.drop("RegionName", axis=1)
    zillow_df = zillow_data_cleaned.T
    zillow_pct_change = zillow_df.pct_change()[zillow_df.index > window_start]
    last_date = zillow_pct_change.index[-1]
    month_gap = (pd.Timestamp(datetime.datetime.now()).to_period('M') -
                 pd.Timestamp(last_date).to_period("M"))
    zillow_pct_now = ((zillow_pct_change.mean() + 1) ** int(month_gap.freqstr[:-1])).reset_index()
    zillow_pct_now.columns = ['RegionName', 'est_pct']
    zillow_value_now = zillow_data_cleaned.merge(zillow_pct_now, on='RegionName', how='left')
    zillow_value_now['current_value'] = zillow_value_now[last_date] * zillow_value_now['est_pct']
    zillow_value_now = zillow_value_now[["RegionName", "current_value"]]
    zillow_value_now.columns = ["zipcode", 'current_value']
    return zillow_value_now


current_value_pipeline = Pipeline(
    [("current_value", FunctionTransformer(current_median, validate=False, kw_args={'window_start': "2012-06"}))])


zillow_cost=current_value_pipeline.fit_transform(zillow_data.copy())

class CalProfit(BaseEstimator, TransformerMixin):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def fit(self, X, Y, y=None):
        return self

    def transform(self, X):
        self.X.zipcode = self.X.zipcode.astype("int")
        # left join airbnb data and zillow data
        revenue_cost = self.X.merge(self.Y, how="left", on="zipcode")
        # group by neighbourhood, sum revenue and square feet
        agg = revenue_cost[~revenue_cost.current_value.isnull()].groupby(
            revenue_cost.columns[neighbourhood_group_cleansed_ix]).agg({
                'current_value':
                'sum',
                revenue_cost.columns[square_feet_ix]:
                'sum'
            }).reset_index()
        # calculate average cost per square feet
        agg['avg_cost_per_square_feet'] = agg.iloc[:, 1] / agg.iloc[:, 2]
        agg = agg[[
            revenue_cost.columns[neighbourhood_group_cleansed_ix],
            'avg_cost_per_square_feet'
        ]]
        # using the assumption that average cost per square feet is constant in each neighbourhood
        # to calculate estimated cost for each house
        revenue_cost=revenue_cost.merge(agg, on=revenue_cost.columns[neighbourhood_group_cleansed_ix])
        revenue_cost['est_cost']=revenue_cost['avg_cost_per_square_feet'] * revenue_cost.iloc[:,square_feet_ix]
        return revenue_cost


class CalEvaluationMatrix(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['break_even_year'] = X['est_cost'] / X['est_yearly_revenue']  # break even years calculation
        X['annual_roi'] = X['est_yearly_revenue'] / X['est_cost'] * 100  # annual ROI calculation
        return X


def top_n_zipcode(X, variable: "['break_even_year', 'annual_roi']", top_n) -> "DataFrame":
    """
    calcualate the median of the selected variable, then get the top n records
    """
    final_roi = X.groupby('zipcode')[[variable]].median().reset_index().sort_values(by=variable)
    return final_roi.head(top_n)[["zipcode", variable]]


profit_pipeline = Pipeline([('cal_profit',
                             CalProfit(airbnb_revenue, zillow_cost)),
                            ('cal_roi_year', CalEvaluationMatrix()),
                            ('top_n_zipcode',
                             FunctionTransformer(top_n_zipcode,
                                                 validate=False,
                                                 kw_args={'variable': 'annual_roi','top_n': 10}))])

# profit_pipeline.fit_transform(airbnb_revenue, zillow_cost)