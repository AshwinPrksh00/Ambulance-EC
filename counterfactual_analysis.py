import numpy as np
import pandas as pd
import os
from neuralprophet import NeuralProphet
import dabest
import holidays
import seaborn
import datetime as dt
import pickle
from scipy.stats import wilcoxon, shapiro
from cliffs_delta import cliffs_delta
import plotly.express as px
import plotly.graph_objects as go
import argparse
from utils.utils import PreProcess, ProcessResults
from utils.trainEach import TrainModel
from utils.trainAll import InferenceTrain
# import optuna

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default=None, help='Path to data file')
parser.add_argument('--model', type=str, default=None, help='Path to model file. If there is no model file, training will occur.')
parser.add_argument('--type', type=str, default='rt', help='Analysis to be performed on which type of data, Values: rt = Response Time, ht = Handover Time, default = rt')

args = parser.parse_args()

#Checking if path to data file is provided
if args.data is not None:
    file_ext = os.path.splitext(args.data)[1]
    if file_ext == '.csv':
        data_df = pd.read_csv(args.data)
    elif file_ext == '.xlsx':
        data_df = pd.read_excel(args.data)
    elif file_ext == '.pkl':
        data_df = pd.read_pickle(args.data)
    else:
        raise Exception('Invalid file format. Please provide a .csv, .xlsx or .pkl file.')
else:
    raise Exception('Please provide a path to the data file.')


#Cleaning the data
data_df = PreProcess(data_df).clean_feed_data(args.type)





#Checking if path to model file is provided
if args.model is not None:
    if os.path.splitext(args.model)[1] is not '.pkl':
        raise Exception('Invalid file format. Please provide a .pkl file.')
    else:
        print('Model File Found, Turning to Inference Mode')