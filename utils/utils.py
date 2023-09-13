import numpy as np
import pandas as pd
from neuralprophet import NeuralProphet
import dabest
import datetime as dt
from cliffs_delta import cliffs_delta


#Creation of class for results processing

class PreProcess:
    def __init__(self, df:pd.DataFrame):
        self.df = df
        pass

    def clean_feed_data(self, type: str):
        #Cleaning of feed data
        self.df['Date'] = pd.to_datetime(self.df['Date'], format="%d/%m/%Y")
        self.df['Day'] = self.df['Date'].dt.day_name()
        if type == 'rt':
            self.df = self.df[['Date', 'Response Time', 'Day']]
            self.df = self.df.rename(columns={'Response Time':'Number of Calls'})
        elif type == 'ht':
            self.df = self.df[['Date', 'Hospital (Mins)', 'Day']]
            self.df = self.df.rename(columns={'Hospital (Mins)':'Number of Calls'})
        else:
            raise Exception('Invalid type. Please specify either rt or ht')
        
        return self.df


class ProcessResults:
    def __init__(self, df:pd.DataFrame):
        self.df = df
        pass

    def calculatePchange(orig_col, new_col, name='', calls=False): #Calculation of Percentage Change
        orig_col.reset_index(inplace=True, drop=True)
        new_col.reset_index(inplace=True, drop=True)
        # weightSum = (orig_col - new_col).median()
        if calls:
            original_sum = orig_col.sum()
            predicted_sum = new_col.sum()
            weightSum = original_sum - predicted_sum
            print(orig_col.sum(), new_col.sum(), weightSum)
            predicted_avg_day = predicted_sum/len(new_col)
            original_avg_day = orig_col.sum()/len(orig_col) 
            pChange = (weightSum/predicted_sum)*100
            print(f'{name} Percentage Change: {pChange}')
            return original_sum, predicted_sum, pChange, predicted_avg_day, original_avg_day
        else:
            weightSum = orig_col.median() - new_col.median()
            predicted_sum = new_col.median()
            predicted_avg_day = predicted_sum/len(new_col)
            original_avg_day = orig_col.sum()/len(orig_col)
            pChange = (weightSum/predicted_sum)*100
            print(f'{name} Percentage Change: {pChange}')
            return orig_col.median(), new_col.median(), pChange, predicted_avg_day, original_avg_day
        
    def calcWMAPE(self): #WMAPE Calculation
        df_temp1 = self.df[(self.df['ds']>=dt.datetime.strptime('2020-01-01',"%Y-%m-%d")) & (self.df['ds']<=dt.datetime.strptime('2020-03-22',"%Y-%m-%d"))]
        df_temp1['weights'] = np.abs(df_temp1['y_y'] - df_temp1['yhat1'])
        weightSum = df_temp1['weights'].sum()
        actualSum = df_temp1['y_y'].sum()
        wmape = weightSum/actualSum
        print(f"Data Range : {self.df['ds'][103]} - {self.df['ds'][len(self.df['ds'])-1]}")
        print(f"WMAPE Range: {df_temp1['ds'][0]} - {df_temp1['ds'][len(df_temp1['ds'])-1]}")
        print(f'WMAPE : ', wmape)
        return wmape
    
    def cohend(self, d1: pd.Series, d2: pd.Series) -> float:

        # calculate the size of samples
        n1, n2 = len(d1), len(d2)

        # calculate the variance of the samples
        s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)

        # calculate the pooled standard deviation
        s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))

        # calculate the means of the samples
        u1, u2 = np.mean(d1), np.mean(d2)

        # return the effect size
        chd_res = (u1 - u2) / s
        print('Effect Size Cohens d: ', chd_res)
        print('---------------------------')

        #Calculate the measure of effect size
        if chd_res <= 0.2 and chd_res >= 0:
            chd_range = 'Very small'
        elif chd_res >=-0.2 and chd_res <= 0:
            chd_range = 'Very small'
        elif chd_res <= 0.5 and chd_res > 0.2:
            chd_range = 'Small'
        elif chd_res >= -0.5 and chd_res < -0.2:
            chd_range = 'Small'
        elif chd_res <= 0.8 and chd_res > 0.5:
            chd_range = 'Medium'
        elif chd_res >= -0.8 and chd_res < -0.5:
            chd_range = 'Medium'
        elif chd_res <= 1.2 and chd_res > 0.8:
            chd_range = 'Large'
        elif chd_res >= -1.2 and chd_res < -0.8:
            chd_range = 'Large'
        elif chd_res > 1.2:
            chd_range = 'Very Large'
        elif chd_res < -1.2:
            chd_range = 'Very Large'
        
        return chd_res, chd_range

    def effect_sizeCI(self, orig_col, new_col, tname=''):
        CI_dat = pd.DataFrame()
        CI_dat['orig'] = orig_col
        CI_dat['new'] = new_col
        dabest_dataframe = dabest.load(CI_dat, idx=('new', 'orig'))
        if tname=='cliffs delta':
            eff_sz = dabest_dataframe.cliffs_delta
            eff_sz_1, es = cliffs_delta(orig_col, new_col)
        elif tname=='cohens_d':
            eff_sz = dabest_dataframe.cohens_d
            eff_sz_1, es = self.cohend(orig_col, new_col)
        else:
            raise Exception('Invalid effect size name')
        
        eff_sz_0 = np.round(np.mean(eff_sz.results['bootstraps'][0]), 2)
        ci_upper, ci_lower = np.round(eff_sz.results['bca_high'][0], 2), np.round(eff_sz.results['bca_low'][0], 2)
        print(f'{tname} CI: [{ci_lower}, {ci_upper}]')
        return eff_sz_0, np.round(eff_sz_1,2), es, ci_lower, ci_upper
    
