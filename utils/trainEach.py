import numpy as np
import pandas as pd
import os
from neuralprophet import NeuralProphet
import holidays
import datetime as dt
import pickle
from scipy.stats import wilcoxon, shapiro
from utils import ProcessResults
# import optuna


#Creation of Training Class
class TrainModel:
    def __init__(self, df, name, emname,  model=None, qr=False, gname=None, epochs=3000, lines=False, holiday=False, plot_new=False, gpu=False, timez=None, call=False) -> None:
        self.df = df
        self.name = name
        self.emname = emname
        self.model = model
        self.qr = qr
        self.gname = gname
        self.epochs = epochs
        self.lines = lines
        self.holiday = holiday
        self.plot_new = plot_new
        self.gpu = gpu
        self.timez = timez
        self.call = call
        pass

    def train_predict(self):
        #List for storing Percentage Change
        p = []
        # Time periods
        till2019 = dt.datetime.strptime('2020-01-01',"%Y-%m-%d")
        valPeriod = dt.datetime.strptime('2020-03-22',"%Y-%m-%d")
        # till2021 = dt.datetime.strptime('2021-12-31',"%Y-%m-%d")
        # data_2020 = df[(df['Date']>=dt.datetime.strptime('2020-01-01',"%Y-%m-%d"))]
        data_2020 = self.df[(self.df['Date']>=till2019)]
        if 'index' in data_2020.columns:
            data_2020.drop('index', inplace=True, axis=1)
        data_2020 = data_2020[['Date', 'Number of Calls', 'Day']]
        data_2020.columns = ['ds', 'y', 'Day']

        #Training Data
        # train_data = df[df['Date']<till2019]
        train_data = self.df[self.df['Date']<till2019]
        train_data = train_data[['Date', 'Number of Calls']]
        train_data.columns = ['ds', 'y']
        
        #Validation Data
        val_data = self.df[(self.df['Date']>=till2019) & (self.df['Date']<=valPeriod)]
        val_data = val_data[['Date', 'Number of Calls']]
        val_data.columns = ['ds', 'y']

        if self.model is None:

            #Model Training
            print('Training Model...')
            # model1 = NeuralProphet(trend_reg_threshold=True, epochs=epochs, batch_size=64, n_forecasts=1004, num_hidden_layers=10, learning_rate=0.01, seasonality_reg=20, ar_reg=0.2 growth='discontinuous')
            quantile_lo, quantile_hi = 0.05, 0.95
            quantiles = [quantile_lo, quantile_hi]
            if self.gpu:
                model1 = NeuralProphet(trend_reg_threshold=True, epochs=self.epochs, batch_size=128, n_forecasts=365, learning_rate=0.01, seasonality_reg=20, ar_reg=0.2, growth='discontinuous',
                                quantiles=quantiles, accelerator='cuda', trainer_config={'accelerator': 'gpu'})
            else:
                model1 = NeuralProphet(trend_reg_threshold=True, epochs=self.epochs, batch_size=128, n_forecasts=365, learning_rate=0.01, seasonality_reg=20, ar_reg=0.2, growth='discontinuous',
                                quantiles=quantiles)

            if self.holiday:
                #Adding Holidays
                ind_holidays = holidays.IND(years=[y for y in range(2016,2020)])
                hol_list = pd.to_datetime(list(ind_holidays.keys()))
                # holidays_india = pd.DataFrame([{'ds':df22_total_normalized['Date'][i], 'event':'holiday_IND'} for i in df22_total_normalized.index if df22_total_normalized['Date'][i] in hol_list])
                # #Add event to model
                model1.add_country_holidays(country_name='IND')
                # model1 = model1.add_events("holiday_IND")
                # # create the data df with events
                # history_df = model1.create_df_with_events(train_data, holidays_india)
                # # fit the model
                # metrics1 = model1.fit(history_df, freq='D', validation_df=val_data)
            metrics1 = model1.fit(train_data, freq='D', validation_df=val_data, progress=None)

            #Saving Model
            if not os.path.exists('Models'):
                os.mkdir('Models')
            if self.timez is not None:
                with open('Models/D-N/Model_'+self.name+'_1722.pkl', 'wb') as f:
                    pickle.dump(model1, f)
            else:
                with open('Models/Model_'+self.name+'_1722.pkl', 'wb') as f:
                    pickle.dump(model1, f)
        else:
            #Reloading Model
            print(f'Reloading Model...{self.model}')
            with open(self.model, 'rb') as f:
                model1 = pickle.load(f)
            model1.restore_trainer()

        #Predicting for 2020
        # if holidays:
        #     data_future = model1.make_future_dataframe(df = history_df, events_df = holidays_india, periods=1004)
        #     forecast = model1.predict(data_future)
        # else:
        data_future = model1.make_future_dataframe(train_data, periods=1096)
        if self.qr:
                method = "cqr"
                alpha = 0.1
                plotting_backend = "matplotlib"

                cqr_df = model1.conformal_predict(
                                                    data_future,
                                                    calibration_df=data_2020[['ds', 'y']],
                                                    alpha=alpha,
                                                    method=method
                                                )
                forecast = cqr_df
        else:
            forecast = model1.predict(data_future)

        #Merge original and predicted data
        df4 = pd.merge(forecast, data_2020, how='left', on='ds')
        #Round the floating values
        df4.yhat1 = np.round(df4.yhat1)


        print("-----------------------------------RESULTS---------------------------------------")

        resultCalc = ProcessResults(df4)
        #WMAPE Calculations
        print("WMAPE Calculation for "+self.name+' '+self.emname)
        resultCalc.calcWMAPE()
        if self.timez is not None:
            print(f'Time: {self.timez}')

        #Copy of predicted dataframe for plotting
        df4_copy = df4.copy(deep=True)
        #Merging predicted data with pre lockdown data
        df4 = df4[['ds', 'yhat1', 'Day']]
        df4.columns = ['Date', 'Number of Calls', 'Day']
        org_data = self.df[self.df['Date']<till2019]
        org_data = org_data[['Date', 'Number of Calls', 'Day']]
        df_merged = pd.concat([org_data, df4])
        # return df_merged

        #Wilcoxon Analysis
        timePeriods = ['2020-03-22', '2020-03-23', '2020-09-30', '2020-10-01', '2021-03-31', '2021-04-01', '2021-09-30', '2021-10-01', '2021-12-20', '2021-12-21', '2022-03-11']
        waves = ['Pre-Wave 1','Wave 1', 'Post Wave 1', 'Wave 2', 'Post Wave 2', 'Wave 3', 'Post Wave 3']

        #For Prewave 1
        # df_w1 = df_merged[(df_merged['Date']<=dt.datetime.strptime(timePeriods[0],"%Y-%m-%d"))]['Number of Calls']
        # df_original = df[(df['Date']<=dt.datetime.strptime(timePeriods[0],"%Y-%m-%d"))]['Number of Calls']
        # res_df_w1 = wilcoxon(df_w1, df_original)
        # print(waves[0])
        # print(res_df_w1)
        iter=1
        for i in range(5):
            df_w1 = df_merged[(df_merged['Date']>=dt.datetime.strptime(timePeriods[iter],"%Y-%m-%d")) & (df_merged['Date']<=dt.datetime.strptime(timePeriods[iter+1],"%Y-%m-%d"))]['Number of Calls']
            df_original = self.df[(self.df['Date']>=dt.datetime.strptime(timePeriods[iter],"%Y-%m-%d")) & (self.df['Date']<=dt.datetime.strptime(timePeriods[iter+1],"%Y-%m-%d"))]['Number of Calls']
            predCI = df4_copy[(df4_copy['ds']>=dt.datetime.strptime(timePeriods[iter],"%Y-%m-%d")) & (df4_copy['ds']<=dt.datetime.strptime(timePeriods[iter+1],"%Y-%m-%d"))][['ds', 'yhat1 5.0% - qhat1', "yhat1 95.0% + qhat1"]]
            predCI.columns = ['Date', 'Lower Bound', 'Upper Bound']
            #Percentage Change
            resP = resultCalc.calculatePchange(df_original, df_w1, waves[i+1], calls=self.call)
            #Shapiro-wilk Test
            _, orig_shap_p = shapiro(df_original)
            _, pred_shap_p = shapiro(df_w1)
            #Checking for normality
            try:
                if orig_shap_p >= 0.05 and pred_shap_p >= 0.05:
                    res_df_w1 = np.nan
                    #Cohens'd effect size
                    efs = resultCalc.effect_sizeCI(df_original, df_w1, tname="cohens_d")
                    es = f'{round(float(efs[1]),2)} {efs[2]}*'
                else:
                    #Cliff's Delta effect size
                    efs = resultCalc.effect_sizeCI(df_original, df_w1, tname="cliffs delta")
                    es = f'{round(float(efs[1]),2)} {efs[2]}'
                    print(waves[i+1] + " Cliffs Delta")
                    print(es)
                    _, res_df_w1 = wilcoxon(df_w1, df_original)

            except:
                print('------------Error--------------')
                print(f'1. Length Mismatch btw df_w1 {len(df_w1)} and df_original {len(df_original)}')
                print('2. Check Cliffs Delta Function')
                res_df_w1 = np.nan
                
                #Wlicoxon Test
                # res_df_w1 = wilcoxon(df_w1, df_original)
                # print(waves[i+1] + " Wilcoxon Analysis")
                # print(res_df_w1)
                # #Effect Size calculation

                # # res_dis, res_jdg = cliffs_delta(df_original, df_w1)
                # res_cliffs = effect_sizeCI(df_original, df_w1)
            print(f"Confidence Interval Predicted: [{predCI['Lower Bound'].median()}, {predCI['Upper Bound'].median()}]")
            if self.timez is not None:
                p.append({
                    'Wave': waves[i+1],
                    'Emergency Type': self.emname,
                    'Time': self.timez,
                    'Actual':resP[0],
                    'Predicted':resP[1],
                    'Percentage Change': resP[2],
                    f'Avg Predicted {self.gname} per Day': resP[3],
                    f'Avg Actual {self.gname} per Day': resP[4],
                    'Confidence Interval - Predicted': [np.round(predCI['Lower Bound'].median(),2), np.round(predCI['Upper Bound'].median(),2)],
                    'Normality Test (Shapiro-Wilk) - Actual': orig_shap_p,
                    'Normality Test (Shapiro-Wilk) - Predicted': pred_shap_p,
                    'Wilcoxon Test': res_df_w1,
                    'Effect Size': es,
                    'Confidence Interval - Effect Size': [efs[3], efs[4]],
                    'Mode': self.name
                })
            else:

                p.append({
                    'Wave': waves[i+1],
                    'Emergency Type': self.emname,
                    'Actual':resP[0],
                    'Predicted':resP[1],
                    'Percentage Change': resP[2],
                    f'Avg Predicted {self.gname} per Day': resP[3],
                    f'Avg Actual {self.gname} per Day': resP[4],
                    'Confidence Interval - Predicted': [np.round(predCI['Lower Bound'].median(),2), np.round(predCI['Upper Bound'].median(),2)],
                    'Normality Test (Shapiro-Wilk) - Actual': orig_shap_p,
                    'Normality Test (Shapiro-Wilk) - Predicted': pred_shap_p,
                    'Wilcoxon Test': res_df_w1,
                    'Effect Size': es,
                    'Confidence Interval - Effect Size': [efs[3], efs[4]],
                    'Mode': self.name
                })
            iter += 2
        
        df_w1 = df_merged[(df_merged['Date']>dt.datetime.strptime(timePeriods[-1],"%Y-%m-%d"))]['Number of Calls']
        df_original = self.df[(self.df['Date']>dt.datetime.strptime(timePeriods[-1],"%Y-%m-%d"))]['Number of Calls']
        predCI = df4_copy[(df4_copy['ds']>dt.datetime.strptime(timePeriods[-1],"%Y-%m-%d"))][['ds', 'yhat1 5.0% - qhat1', "yhat1 95.0% + qhat1"]]
        predCI.columns = ['Date', 'Lower Bound', 'Upper Bound']
        # print(df_merged[(df_merged['Date']>dt.datetime.strptime(timePeriods[-1],"%Y-%m-%d"))]['Date'].tail(1), df[(df['Date']>dt.datetime.strptime(timePeriods[-1],"%Y-%m-%d"))]['Date'].tail(1))
        #Percentage Change
        resP = resultCalc.calculatePchange(df_original, df_w1, waves[-1], calls=self.call)
        print(df_w1.shape, df_original.shape)
        #Shapiro-Wilk Test
        _, orig_shap_p = shapiro(df_original)
        _, pred_shap_p = shapiro(df_w1)
        #Checking for normality
        if orig_shap_p >= 0.05 and pred_shap_p >= 0.05:
            res_df_w1 = np.nan
            #Cohens'd effect size
            efs = resultCalc.effect_sizeCI(df_original, df_w1, tname="cohens_d")
            es = f'{round(float(efs[1]),2)} {efs[2]}*'
        else:
            #Wilcoxon Test
            _, res_df_w1 = wilcoxon(df_w1, df_original)
            #Cliff's Delta effect size
            efs = resultCalc.effect_sizeCI(df_original, df_w1, tname="cliffs delta")
            es = f'{round(float(efs[1]),2)} {efs[2]}'
            print(waves[-1] + " Cliffs Delta")
            print(es)
        #Wlicoxon Test
        # res_df_w1 = wilcoxon(df_w1, df_original)
        # print(waves[-1] + " Wilcoxon Analysis")
        # print(res_df_w1)
        # res_cliffs = cliffs_deltaCI(df_original, df_w1)
        print(f"Confidence Interval Predicted: [{predCI['Lower Bound'].median()}, {predCI['Upper Bound'].median()}]")

        if self.timez is not None:
            p.append({
                'Wave': waves[-1],
                'Emergency Type': self.emname,
                'Time': self.timez,
                'Actual':resP[0],
                'Predicted':resP[1],
                'Percentage Change': resP[2],
                f'Avg Predicted {self.gname} per Day': resP[3],
                f'Avg Actual {self.gname} per Day': resP[4],
                'Confidence Interval - Predicted': [np.round(predCI['Lower Bound'].median(),2), np.round(predCI['Upper Bound'].median(),2)],
                'Normality Test (Shapiro-Wilk) - Actual': orig_shap_p,
                'Normality Test (Shapiro-Wilk) - Predicted': pred_shap_p,
                'Wilcoxon Test': res_df_w1,
                'Effect Size': es,
                'Confidence Interval - Effect Size': [efs[3], efs[4]],
                'Mode': self.name
            })
        else:
            p.append({
                'Wave': waves[-1],
                'Emergency Type': self.emname,
                'Actual':resP[0],
                'Predicted':resP[1],
                'Percentage Change': resP[2],
                f'Avg Predicted {self.gname} per Day': resP[3],
                f'Avg Actual {self.gname} per Day': resP[4],
                'Confidence Interval - Predicted': [np.round(predCI['Lower Bound'].median(),2), np.round(predCI['Upper Bound'].median(),2)],
                'Normality Test (Shapiro-Wilk) - Actual': orig_shap_p,
                'Normality Test (Shapiro-Wilk) - Predicted': pred_shap_p,
                'Wilcoxon Test': res_df_w1,
                'Effect Size': es,
                'Confidence Interval - Effect Size': [efs[3], efs[4]],
                'Mode': self.name
            })
        
        df_percent = pd.DataFrame(p)
        # df_percent.to_csv('Statistics Result.csv') #Saving the results
        return df_percent

