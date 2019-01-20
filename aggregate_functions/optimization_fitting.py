import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

class ModelFitting():
    def __init__(self, output_prediction_A, output_prediction_B, output_prediction_C):
        super(ModelFitting, self).__init__()
        self.output_prediction_A = output_prediction_A
        self.output_prediction_B = output_prediction_B
        self.output_prediction_C = output_prediction_C
        
        self.fitting_of_prediction_A()
        self.fitting_of_prediction_B()
        self.fitting_of_prediction_C()

    def prediction_fitting_function(self, file, item):
        """Generic function for getting the error rates, and plotting of function"""
        predicton_fitting = (
            pd.read_csv(file)
            .pipe(lambda x:x.assign(error = x.predicted_values - x.amount))
            .sort_values(['date_id'],ascending=True)
        )

        # calculate RMSE
        self.rmse = np.sqrt(predicton_fitting.error.sum()/len(predicton_fitting))
        self.average_values = predicton_fitting.amount.mean()
        self.error_rates = self.rmse/self.average_values
        print("average quantity demanded for " + item + " across time series: " + str(round(self.average_values,2)))
        print("RMSE for " + item + ": " + str(round(self.rmse,2)))
        print("error margin for " + item + ": " + str(round(self.error_rates,2)))

        # plotting of item
        plt.figure(figsize=(20,10))
        plt.plot(predicton_fitting.date_id, predicton_fitting.amount, label = 'actual')
        plt.plot(predicton_fitting.date_id, predicton_fitting.predicted_values, label = 'predicted')
        plt.xticks(rotation=45)
        plt.legend(loc='upper right')
        plt.title("item " + item + ", actual against predicted")

        plt.show()
        return self.average_values

    def fitting_of_prediction_A(self):
        """Getting the error margin of the demand prediction A"""
        print("Start of item A:")
        self.prediction_fitting_A = self.prediction_fitting_function(self.output_prediction_A, 'A')


    def fitting_of_prediction_B(self):
        """Getting the error margin of the demand prediction B"""
        print("Start of item B:")
        self.prediction_fitting_B = self.prediction_fitting_function(self.output_prediction_B, 'B')

    def fitting_of_prediction_C(self):
        """Getting the error margin of the demand prediction B"""
        print("Start of item C:")
        self.prediction_fitting_C = self.prediction_fitting_function(self.output_prediction_C, 'C')