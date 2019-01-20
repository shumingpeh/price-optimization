import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

class RegressionModel():
    def __init__(self, itemA_file, itemB_file, itemC_file):
        super(RegressionModel, self).__init__()
        self.itemA_file = itemA_file
        self.itemB_file = itemB_file
        self.itemC_file = itemC_file
        
        self.regression_item_A()
        self.regression_item_B()
        self.regression_item_C()

    def regression_item_A(self):
        """Run the regression for itemA"""
        self.itemA = pd.read_csv(self.itemA_file)

        # this selects what variables to choose
        X = self.itemA[['is_weekday','is_weekend','is_holiday','previous_day_demand','weighted_price_usd_A',
                             'weighted_price_usd_B','weighted_price_usd_C','is_promotion_B','is_promotion_C']]
        X = sm.add_constant(X)
        y = self.itemA[['amount']]

        # fitting of model
        model = sm.OLS(y, X).fit()
        self.model_params_A = model.params

        # getting out the prediction results of item A
        output_prediction = (
            self.itemA
            .merge(
                pd.DataFrame(model.predict(X))
                .pipe(lambda x:x.assign(date_id = self.itemA.date_id))
                .rename(columns={0:"predicted_values"})
                ,how='inner',on=['date_id']
            )
            [['date_id','amount','predicted_values']]
        )
        output_prediction.to_csv('data/output_prediction_A.csv',index=False)

    def regression_item_B(self):
        """Run the regression for itemB"""
        self.itemB = pd.read_csv(self.itemB_file)

        # this selects what variables to choose
        X = self.itemB[['is_weekday','is_weekend','is_holiday','previous_demand','weighted_price_usd_A',
                             'weighted_price_usd_B','weighted_price_usd_C','is_promotion_A','is_promotion_C']]
        X = sm.add_constant(X)
        y = self.itemB[['amount']]

        # fitting of model
        model = sm.OLS(y, X).fit()
        self.model_params_B = model.params

        # getting out the prediction results of item B
        output_prediction = (
            self.itemB
            .merge(
                pd.DataFrame(model.predict(X))
                .pipe(lambda x:x.assign(date_id = self.itemB.date_id))
                .rename(columns={0:"predicted_values"})
                ,how='inner',on=['date_id']
            )
            [['date_id','amount','predicted_values']]
        )
        output_prediction.to_csv('data/output_prediction_B.csv',index=False)

    def regression_item_C(self):
        """Run the regression for itemC"""
        self.itemC = pd.read_csv(self.itemC_file)

        # this selects what variables to choose
        X = self.itemC[['is_weekday','is_weekend','is_holiday','previous_demand','weighted_price_usd_A',
                             'weighted_price_usd_B','weighted_price_usd_C','is_promotion_B','is_promotion_A']]
        X = sm.add_constant(X)
        y = self.itemC[['amount']]

        # fitting of model
        model = sm.OLS(y, X).fit()
        self.model_params_C = model.params 

        # getting out the prediction results of item C
        output_prediction = (
            self.itemC
            .merge(
                pd.DataFrame(model.predict(X))
                .pipe(lambda x:x.assign(date_id = self.itemC.date_id))
                .rename(columns={0:"predicted_values"})
                ,how='inner',on=['date_id']
            )
            [['date_id','amount','predicted_values']]
        )
        output_prediction.to_csv('data/output_prediction_C.csv',index=False)
        
