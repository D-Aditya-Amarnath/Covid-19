import os
import pandas as pd
import numpy as np
from sklearn import metrics
from statsmodels.tsa.api import ExponentialSmoothing,Holt
from sklearn.model_selection import ParameterGrid
from timeit import default_timer as timer
import multiprocessing
import pprint
import warnings
import concurrent.futures



def model_func(train,test,b):
    trend = b.get("trend")
    seasonal = b.get("seasonal")
    seasonal_periods = b.get('seasonal_periods')
    smoothing_level=b.get('smoothing_level')
    smoothing_slope = b.get('smoothing_slope')
    damping_slope = b.get('damping_slope')
    damped=b.get('damped')
    smoothing_seasonal = b.get('smoothing_seasonal')
    #use_boxcox = b.get('use_boxcox')
    #remove_bias = b.get('remove_bias')
    #use_basinhopping = b.get('use_basinhopping')
    RMSE,r2=0,0
    fitTES = ExponentialSmoothing(train,trend=trend,seasonal=seasonal,seasonal_periods=seasonal_periods,damped_trend=damped).fit(
        smoothing_level=smoothing_level,
        smoothing_slope=smoothing_slope,
        damping_slope=damping_slope,
        smoothing_seasonal = smoothing_seasonal,
        #use_boxcox=use_boxcox,
        optimized = False)
    fitTES_pred = fitTES.forecast(30)
    fitTES_pred.replace([np.inf,-np.inf],np.nan,inplace=True)
    fitTES_pred.dropna(inplace=True)
    if len(fitTES_pred) == len(test):    
        RMSE = np.sqrt(metrics.mean_squared_error(test,fitTES_pred))
        print(f'RMSE is {RMSE}')
        r2 = metrics.r2_score(test,fitTES_pred)
    else:
        RMSE = np.nan
        r2 = np.nan
    b['RMSE'] = RMSE
    b['r2'] = r2
    return b

data_doc = "covid_time_series.csv"
covid_data = pd.read_csv(data_doc,parse_dates=True,index_col=['Date'])
covid_data = covid_data[covid_data.index>"2020-03-01"]
covid_data.index.freq="D"
train = covid_data['Daily Confirmed'].iloc[:-30].copy()
test = covid_data['Daily Confirmed'].iloc[-30:].copy()
param_grid = {
    'trend':['add','mul'],
    'seasonal':['add','mul'],
    'seasonal_periods':[7,14],
    'smoothing_level':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
    'smoothing_slope':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
    'smoothing_seasonal':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
    'damping_slope':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
    'damped':[True,False],
    #'use_boxcox':[True,False],
    #'remove_bias':[True,False],
    #'use_basinhopping':[True,False]
}
warnings.filterwarnings('ignore')
pg = list(ParameterGrid(param_grid=param_grid))
start = timer()
rest = []

with multiprocessing.Pool(processes=10) as pool:
    results = [pool.apply_async(model_func,args=(train,test,b)) for _,b in enumerate(pg)]
    
    for r in results:
        rest.append(r.get())
"""for a,b in enumerate(pg):
    results.append(model_func(train,test,b))"""
end = timer()
print(f'Total time taken for model function to run : {(end-start)}')
print(len(rest))
df_results_TES = pd.DataFrame(columns=['trend','seasonal','damped','seasonal_periods','smoothing_level',
                                       'smoothing_slope','smoothing_seasonal','damping_slope',
                                       'RMSE','r2'
                                       ])
for r in rest:
    df_results_TES = df_results_TES.append(r,ignore_index=True)
df_results_TES.to_csv('model_parameters6.csv')

end2 = timer()
print(f'Total time taken to run : {(end2-start)}')

