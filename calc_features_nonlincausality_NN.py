#%%
import numpy as np
import pandas as pd
from biosppy.signals import tools
from scipy import interpolate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ARDRegression, BayesianRidge, HuberRegressor, TheilSenRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import gc
from nonlincausality import calculate_causality
from nonlincausality import nonlincausalityNN

#%%

def calc_causality_ml(
    df_gc, 
    maxlag, 
    causality, 
    p_values, 
    id,
):
    learning_rate = 1e-5
    batch_size_num=64
    scaler = StandardScaler().fit(df_gc[:int(len(df_gc)*0.70)])
    x = scaler.transform(df_gc[:int(len(df_gc)*0.70)])
    x_val = scaler.transform(df_gc[int(len(df_gc)*0.70):int(len(df_gc)*0.85)])
    x_test = scaler.transform(df_gc[int(len(df_gc)*0.85):])
    res = nonlincausalityNN(
        x=x,
        NN_config=['d','dr','d','dr'],
        NN_neurons=[20,0.05,20,0.05],
        epochs_num=100,
        learning_rate=learning_rate,
        batch_size_num=batch_size_num,
        maxlag=[maxlag],
        x_test=x_test,
        x_val=x_val,
        z=[],
        z_test=[],
        z_val=[],
        plot=False,
        run=3,
    )
    causality[id] = calculate_causality(res[maxlag].best_errors_X,res[maxlag].best_errors_XY)
    p_values[id] = res[maxlag].p_value


    return causality, p_values, res
    
#%%
maxlag = 50

features_all = []
p_values_resp_rr={}
p_values_rr_resp={}
causality_resp_rr={}
causality_rr_resp={}

for file_name in os.listdir('data/supine'):
    if 'csv' in file_name:
        causality_resp_rr[file_name]=0
        causality_rr_resp[file_name]=0

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
        idx=0
        while causality_resp_rr[file_name]==0 and idx<3:
            causality_resp_rr, p_values_resp_rr, res_resp_rr_nn = calc_causality_ml(df_gc_rr_resp, maxlag, causality_resp_rr, p_values_resp_rr, file_name)
            idx+=1
        idx=0
        while causality_rr_resp[file_name]==0 and idx<3:
            causality_rr_resp, p_values_rr_resp, res_rr_resp_nn = calc_causality_ml(df_gc_resp_rr, maxlag, causality_rr_resp, p_values_rr_resp, file_name)
            idx+=1
        res_resp_rr = pd.DataFrame(causality_resp_rr,index=[f'Resp_RR_NN']).T.reset_index()
        res_rr_resp = pd.DataFrame(causality_rr_resp,index=[f'RR_Resp_NN']).T.reset_index()

        df_features = pd.merge(res_rr_resp, res_resp_rr,on='index')
        df_features.to_csv('nonlinGC_NN.csv')
        gc.collect()
# %%
