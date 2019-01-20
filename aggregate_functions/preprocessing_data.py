import pandas as pd
import numpy as np
import os

class PreprocessingData():
    def __init__(self, rawdata_file, indo_public_holiday_file):
        super(PreprocessingData, self).__init__()
        self.rawdata_file = rawdata_file
        self.indo_public_holiday_file = indo_public_holiday_file
        
        # reading the rawdata file to pandas, and converting the date to datetime
        self.rawdata = (
            pd.read_csv(self.rawdata_file)
            .pipe(lambda x:x.assign(date_id = pd.to_datetime(x.date_id)))
        )

        # reading the public holiday to pandas, and converting the date to datetime
        self.raw_data_indo_public_holiday = (
            pd.read_csv(self.indo_public_holiday_file)
            .pipe(lambda x:x.assign(date = pd.to_datetime(x.date)))
        )
        
        self.overall_retention_combine_df = None
        
        self.transform_data_to_unknown()
        self.include_predictors_for_regression()
        self.structure_data_for_item_A()
        self.structure_data_for_item_B()
        self.structure_data_for_item_C()
        self.regression_data_for_item_A()
        self.regression_data_for_item_B()
        self.regression_data_for_item_C()
        

    def transform_data_to_unknown(self):
        """Transform the original data to be unknown and annoymous"""
        self.rawdata_processed = (
            self.rawdata
            .query("itemid == 660977713 | itemid == 817055263 | itemid == 82223154")
            # change up the itemid
            .pipe(lambda x:x.assign(itemid = np.where(x.itemid == 82223154,'A',np.where(x.itemid == 660977713,'B','C'))))
            # change up orderid
            .pipe(lambda x:x.assign(orderid = np.round((x.orderid *2 + 10)/5,0).astype(np.int64)))
            # change up price_usd
            .pipe(lambda x:x.assign(discount_amount = x.price_usd/x.price_before_discount_usd))
            .pipe(lambda x:x.assign(price_before_discount_usd = x.price_before_discount_usd * 5 + 1.1))
            .pipe(lambda x:x.assign(price_usd = x.price_before_discount_usd * x.discount_amount))
            .drop(['discount_amount'],1)
            .pipe(lambda x:x.assign(is_discount = np.where(x.price_usd < x.price_before_discount_usd,x.amount,0)))
            .pipe(lambda x:x.assign(date_id = pd.to_datetime(x.date_id)))
        )

    def include_predictors_for_regression(self):
        """Include predictors for regression"""
        self.include_predictors_processed = (
            self.rawdata_processed
            # checking if the date is on a weekday
            .pipe(lambda x:x.assign(is_weekday = np.where(x.date_id.dt.dayofweek < 5,1,0)))
            # checking if the date is on a weekend
            .pipe(lambda x:x.assign(is_weekend = np.where(x.date_id.dt.dayofweek < 5,0,1)))
            # is public holiday included
            .merge(self.raw_data_indo_public_holiday,how='left',left_on=['date_id'],right_on=['date'])
            .fillna(0)
            .drop(['date'],1)
        )

    def get_item_prices_and_previous_demand(self,df,item):
        """Generic function to get item prices and previous demand"""
        # this gets the total demand of the item on each day
        get_totaldemand_of_item = (
            df
            .query("itemid == '" + item + "'")
            .sort_values(['date_id','itemid'])
            .groupby(['date_id','itemid'])
            .agg({"amount":"sum"})
            .reset_index()
            .rename(columns={"amount":"total_item"})
            [['date_id','itemid','total_item']]
        )

        # this does some data manipulation to the prices
        combine_total_demand_of_item = (
            df
            # merge the total demand of the item on any given day
            .merge(get_totaldemand_of_item, how='inner',on=['date_id','itemid'])
            # given there could be many prices of items being sold on any day, the weighted average of the price will be used instead
            .pipe(lambda x:x.assign(weighted_price_usd = x.amount/x.total_item * x.price_usd))
            .groupby(['date_id','itemid','is_weekday','is_weekend','is_holiday'])
            # gets the total demand, weighted average price
            .agg({"amount":"sum","weighted_price_usd":"sum","is_discount":"sum"})
            .reset_index()
            # this gets the previous day quantity demanded, if there is no data, it will be 0
            .pipe(lambda x:x.assign(previous_day_demand = x.amount.shift(1)))
            .fillna(0)
            # check if there is a promotion for the item, if >=50% of the amount sold is on discount, it will be considered as promotion
            .pipe(lambda x:x.assign(is_promotion = np.where(x.is_discount/x.amount >= 0.5,1,0)))
            .drop(['is_discount'],1)
        )

        return combine_total_demand_of_item


    def structure_data_for_item_A(self):
        """Structure data for item A"""
        self.total_demand_A = self.get_item_prices_and_previous_demand(self.include_predictors_processed,'A')
        self.total_demand_A = self.total_demand_A.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
        self.lean_version_total_demand_A = (
            self.total_demand_A
            [['date_id','is_promotion','weighted_price_usd']]
            .rename(columns={"is_promotion":"is_promotion_A","weighted_price_usd":"weighted_price_usd_A"})
        )

    def structure_data_for_item_B(self):
        """Structure data for item B"""
        self.total_demand_B = self.get_item_prices_and_previous_demand(self.include_predictors_processed,'B')
        self.total_demand_B = self.total_demand_B.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
        self.lean_version_total_demand_B = (
            self.total_demand_B
            [['date_id','is_promotion','weighted_price_usd']]
            .rename(columns={"is_promotion":"is_promotion_B","weighted_price_usd":"weighted_price_usd_B"})
        )
        
    def structure_data_for_item_C(self):
        """Structure data for item C"""
        self.total_demand_C = self.get_item_prices_and_previous_demand(self.include_predictors_processed,'C')
        self.total_demand_C = self.total_demand_C.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
        self.lean_version_total_demand_C = (
            self.total_demand_C
            [['date_id','is_promotion','weighted_price_usd']]
            .rename(columns={"is_promotion":"is_promotion_C","weighted_price_usd":"weighted_price_usd_C"})
        )

    def regression_data_for_item_A(self):
        """Get all the regression data for item A"""
        regression_data_A = (
            self.total_demand_A
            .merge(self.lean_version_total_demand_B, how='left',on=['date_id'])
            .merge(self.lean_version_total_demand_C, how='left',on=['date_id'])
        )
        regression_data_A.to_csv("data/regression_data_A.csv",index=False)

    def regression_data_for_item_B(self):
        """Get all the regression data for item A"""
        regression_data_B = (
            self.total_demand_B
            .merge(self.lean_version_total_demand_A, how='left',on=['date_id'])
            .merge(self.lean_version_total_demand_C, how='left',on=['date_id'])
        )
        regression_data_B.to_csv("data/regression_data_B.csv",index=False)

    def regression_data_for_item_C(self):
        """Get all the regression data for item A"""
        regression_data_C = (
            self.total_demand_C
            .merge(self.lean_version_total_demand_A, how='left',on=['date_id'])
            .merge(self.lean_version_total_demand_B, how='left',on=['date_id'])
        )
        regression_data_C.to_csv("data/regression_data_C.csv",index=False)



