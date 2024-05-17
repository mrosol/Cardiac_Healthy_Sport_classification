#%%
import numpy as np
import pandas as pd
from biosppy.signals import tools
from scipy import interpolate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ARDRegression, BayesianRidge, HuberRegressor, TheilSenRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import gc
from nonlincausality import nonlincausality_sklearn, calculate_causality
from nonlincausality import nonlincausalityNN
import os

#%%

def calc_causality_ml(
    df_gc_rr_resp, 
    df_gc_resp_rr, 
    sklearn_model, 
    params, maxlag, 
    causality_resp_rr, 
    p_values_resp_rr, 
    causality_rr_resp, 
    p_values_rr_resp,
    id,
):
    scaler = StandardScaler().fit(df_gc_rr_resp[:int(len(df_gc_rr_resp)*0.70)])
    x = scaler.transform(df_gc_rr_resp[:int(len(df_gc_rr_resp)*0.70)])
    x_val = scaler.transform(df_gc_rr_resp[int(len(df_gc_rr_resp)*0.70):int(len(df_gc_rr_resp)*0.85)])
    x_test = scaler.transform(df_gc_rr_resp[int(len(df_gc_rr_resp)*0.85):])
    res_resp_rr = nonlincausality_sklearn(
        x=x,
        sklearn_model=sklearn_model,
        maxlag=[maxlag],
        params=params,
        x_test=x_test,
        x_val=x_val,
        z=[],
        z_test=[],
        z_val=[],
        plot=False,
    )
    causality_resp_rr[id] = calculate_causality(res_resp_rr[maxlag].best_errors_X,res_resp_rr[maxlag].best_errors_XY)
    p_values_resp_rr[id] = res_resp_rr[maxlag].p_value

    scaler = StandardScaler().fit(df_gc_resp_rr[:int(len(df_gc_resp_rr)*0.70)])
    x = scaler.transform(df_gc_resp_rr[:int(len(df_gc_resp_rr)*0.70)])
    x_val = scaler.transform(df_gc_resp_rr[int(len(df_gc_resp_rr)*0.70):int(len(df_gc_resp_rr)*0.85)])
    x_test = scaler.transform(df_gc_resp_rr[int(len(df_gc_resp_rr)*0.85):])
    res_rr_resp = nonlincausality_sklearn(
        x=x,
        sklearn_model=sklearn_model,
        maxlag=[maxlag],
        params=params,
        x_test=x_test,
        x_val=x_val,
        z=[],
        z_test=[],
        z_val=[],
        plot=False,
    )
    causality_rr_resp[id] = calculate_causality(res_rr_resp[maxlag].best_errors_X,res_rr_resp[maxlag].best_errors_XY)
    p_values_rr_resp[id] = res_rr_resp[maxlag].p_value

    return causality_resp_rr, p_values_resp_rr, causality_rr_resp, p_values_rr_resp
    
#%%
maxlag = 50

### MODELS

# sklearn_model=ARDRegression
# params = {
#     'alpha_1':[1e-6,1e-4,1e-2],
#     'alpha_2':[1e-6,1e-4,1e-2],
# }
# sklearn_model=BayesianRidge
# params = {
#     'alpha_1':[1e-6,1e-4,1e-2],
#     'alpha_2':[1e-6,1e-4,1e-2],
# }

# sklearn_model=HuberRegressor
# params = {
#     'epsilon':[2,1.35,1],
#     'alpha':[1e-6,1e-4,1e-2],
# }
# TheilSenRegressor
# sklearn_model=TheilSenRegressor
# params = {
#     'epsilon':[2,1.35,1],
#     'random_state':[123],
# }

# sklearn_model=RandomForestRegressor
# params = {
#     'n_estimatorsint':[100,200,300],
#     'max_depth':[3,5,None],
# }

# sklearn_model=SVR
# params = {
#     'kernel':['poly', 'rbf'],
#     'C':[3,5,None],
# }

params_all = {
    'ARDRegression':{
        'alpha_1':[1e-6,1e-4,1e-2],
        'alpha_2':[1e-6,1e-4,1e-2],
    },
    'BayesianRidge':{
        'alpha_1':[1e-6,1e-4,1e-2],
        'alpha_2':[1e-6,1e-4,1e-2],
    },
    'HuberRegressor':{
        'epsilon':[2,1.35,1],
        'alpha':[1e-6,1e-4,1e-2],
    },
    'TheilSenRegressor': {
        'max_iter':[300]
    },
    'RandomForestRegressor':{
        'n_estimators':[10,25,50],
        'max_depth':[3,5,None],
    },
    'SVR':{
        'kernel':['poly', 'rbf'],
        # 'degree':[3,5],
        'C':[0.01,0.1,1], 
        'epsilon':[0.01,0.1,1.]
    },
    'GaussianProcessRegressor':{
        'normalize_y':[False]
    },
    'GradientBoostingRegressor':{
        'learning_rate':[0.1, 0.01]
    }

}

for sklearn_model in [
    # BayesianRidge, 
    # HuberRegressor, 
    # TheilSenRegressor, 
    # RandomForestRegressor,
    # SVR,
    # GaussianProcessRegressor,
    # GradientBoostingRegressor
    ARDRegression
    ]:
    params = params_all[sklearn_model.__name__]


    features_all = []
    p_values_resp_rr={}
    p_values_rr_resp={}
    causality_resp_rr={}
    causality_rr_resp={}

    features_all = []
    for file_name in os.listdir('data/supine'):
        if 'csv' in file_name:
            print(f'{sklearn_model.__name__} - {file_name}')
            df = pd.read_csv(f'data/supine/{file_name}',index_col=0,sep=',')
            df = df[~df['ECG'].isnull()].reset_index(drop=True)

            rr = df['RR_INTERVALS'].dropna()
            rr_outliers = df['RR_INTERVALS_OUTLIERS'].dropna()
            rr = rr[~rr.isin(rr_outliers)]

            # Define your sampling rate
            sampling_rate = 250  # Replace with the actual sampling rate of your signal

            # Convert breaths per minute to frequency (Hz)
            low_freq = 3 / 60  # 3 breaths per minute in Hz
            high_freq = 40 / 60  # 40 breaths per minute in Hz

            # Apply bandpass filter
            resp_clean, _, _ = tools.filter_signal(signal=df['RESP'],
                                                        ftype='butter',
                                                        band='bandpass',
                                                        order=2,
                                                        frequency=[low_freq, high_freq],
                                                        sampling_rate=sampling_rate)
            resp_clean = -resp_clean

            ### CAUSAL/INFORMATION DOMAIN

            resp_clean = pd.Series(-resp_clean,index=df['RESP'].index)
            rr = df['RR_INTERVALS']
            rr_outliers = df['RR_INTERVALS_OUTLIERS']
            rr[~rr_outliers.isnull()] = None

            rr_inter = rr.interpolate(method='cubic').dropna()
            resp_clean = pd.Series(resp_clean[rr_inter.index[0]:rr_inter.index[-1]+1],index=rr_inter.index)
            rr_inter_short = rr_inter[::10]
            resp_clean_short = resp_clean[::10]
            # Resp in 1st column RR in 2nd
            df_gc_resp_rr = np.array([resp_clean_short.values,rr_inter_short.values]).T
            # RR in 1st column Resp in 2nd
            df_gc_rr_resp = np.array([rr_inter_short.values,resp_clean_short.values]).T

            causality_resp_rr, p_values_resp_rr, causality_rr_resp, p_values_rr_resp = calc_causality_ml(df_gc_rr_resp, df_gc_resp_rr, sklearn_model, params, maxlag, causality_resp_rr, p_values_resp_rr, causality_rr_resp, p_values_rr_resp, file_name)
            gc.collect()

    res_resp_rr = pd.DataFrame(causality_resp_rr,index=[f'Resp_RR_{sklearn_model.__name__}']).T.reset_index()
    res_rr_resp = pd.DataFrame(causality_rr_resp,index=[f'RR_Resp_{sklearn_model.__name__}']).T.reset_index()

    df_features = pd.merge(res_rr_resp, res_resp_rr,on='index')
    df_features = df_features.rename(columns={'index':'ID'})

    # df_features['ID'] = [f'HH{file_id[-7:-4]}' for file_id in df_features['ID']]
    df_features.to_csv(f'feature_store/feat_{sklearn_model.__name__}.csv')
# %%
# np.sum(df_features.iloc[:,1:]>0,axis=0)