import numpy as np
import pandas as pd
import os
from neuralprophet import NeuralProphet
import holidays
import datetime as dt
import pickle
from scipy.stats import wilcoxon, shapiro
from utils import ProcessResults
from trainEach import TrainModel

class InferenceTrain:

    def __init__(self, df, name, model=None, qr=False, gname=None, epochs=3000, lines=False, holiday=False, plot_new=False, gpu=False, dn=False, calls=False) -> None:
        self.df = df
        self.name = name
        self.model = model
        self.qr = qr
        self.gname = gname
        self.epochs = epochs
        self.lines = lines
        self.holiday = holiday
        self.plot_new = plot_new
        self.gpu = gpu
        self.dn = dn
        self.calls = calls
        pass

    def getFullCA(self):
        # Full Counterfactual Analysis for specified group
        tdat, fdat = pd.DataFrame(), pd.DataFrame()
        cols = ['Pregnancy related', 'Trauma (Vehicular)', 'Acute Abdomen','Cardiac/Cardio Vascular', 'Respiratory', 'Trauma (non Vehicular)']
        cols_short = ['preg', 'trauma', 'act_abd', 'cardio', 'respir', 'trauma_non']
        for i,j in zip(cols, cols_short):
            if self.dn:
                for l in ['Day', 'Night']:
                    feed_df = self.df[(self.df['EMERGENCY TYPE']==i) & (self.df['Day/Night']==l)]
                    feed_df['Day'] = feed_df['Date'].dt.day_name()
                    feed_df = feed_df[['Date', 'Number of Calls', 'Day']]
                    if self.model:
                        res_model = TrainModel(feed_df, name=self.name+'_'+j+'_'+l, emname=i, model=f'Models/D-N/Model_{self.name}_{j}_{l}_1722.pkl', qr=self.qr, gname=self.gname, epochs=self.epochs, lines=self.lines, holiday=self.holiday, plot_new=self.plot_new, gpu=self.gpu, timez=l, call=self.calls)
                        res_df = res_model.train_predict()
                    else:
                        res_model = TrainModel(feed_df, name=self.name+'_'+j+'_'+l, emname=i, model=None, qr=self.qr, gname=self.gname, epochs=self.epochs, lines=self.lines, holiday=self.holiday, plot_new=self.plot_new, gpu=self.gpu, timez=l, call=self.calls)
                        res_df = res_model.train_predict()
                    
                    tdat = tdat.append(res_df, ignore_index=True)
            else:
                feed_df = self.df[self.df['EMERGENCY TYPE']==i]
                feed_df['Day'] = feed_df['Date'].dt.day_name()
                feed_df = feed_df[['Date', 'Number of Calls', 'Day']]
                if self.model:
                    res_model = TrainModel(feed_df, name=self.name+'_'+j, emname=i, model=f'Models/D-N/Model_{self.name}_{j}_{l}_1722.pkl', qr=self.qr, gname=self.gname, epochs=self.epochs, lines=self.lines, holiday=self.holiday, plot_new=self.plot_new, gpu=self.gpu, timez=l, call=self.calls)
                    res_df = res_model.train_predict()
                else:
                    res_model = TrainModel(feed_df, name=self.name+'_'+j, emname=i, model=None, qr=self.qr, gname=self.gname, epochs=self.epochs, lines=self.lines, holiday=self.holiday, plot_new=self.plot_new, gpu=self.gpu, timez=l, call=self.calls)
                    res_df = res_model.train_predict()
                tdat = tdat.append(res_df, ignore_index=True)

        waves = ['Wave 1', 'Post Wave 1', 'Wave 2', 'Post Wave 2', 'Wave 3', 'Post Wave 3']
        dat_grp = tdat.groupby('Wave')
        for i in waves:
            pdat = dat_grp.get_group(i)
            fdat = fdat.append(pdat, ignore_index=True)
            fdat = fdat.append(pd.Series([np.nan]*len(pdat.columns)), ignore_index=True)
        
        fdat.to_csv('Statistics Result Time.csv')
        return fdat