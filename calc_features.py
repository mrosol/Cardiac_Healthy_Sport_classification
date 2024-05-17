#%%
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import neurokit2 as nk
from biosppy.signals import tools
from scipy import interpolate
from utils import *
from neurokit2.complexity.utils_complexity_symbolize import complexity_symbolize
from neurokit2.complexity.utils_complexity_embedding import complexity_embedding
from arch.unitroot import KPSS
from arch.unitroot import PhillipsPerron
#%%

features_all = []
nonstationary_list = []
for file_name in os.listdir('data/supine'):
    if 'csv' in file_name:
        df = pd.read_csv(f'data/supine/{file_name}',index_col=0,sep=',')
        df = df[~df['ECG'].isnull()].reset_index(drop=True)

        rr = df['RR_INTERVALS'].dropna()
        rr_outliers = df['RR_INTERVALS_OUTLIERS'].dropna()
        rr = rr[~rr.isin(rr_outliers)]

        hrv = nk.hrv(nk.intervals_to_peaks(rr*250),sampling_rate=250)

        sym_dyn = calc_symdyn(rr.dropna().values)

        ### RESP
        
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

        resp_features = calc_resp_features(resp_clean)

        ## CAUSAL/INFORMATION DOMAIN

        # GC
        # resp_clean = pd.Series(-resp_clean,index=df['RESP'].index)
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

        res = PhillipsPerron(rr.dropna().values)
        if res.pvalue>=0.05:
            nonstationary_list.append(file_name)
            print(file_name)

        # RR->Resp (2nd column->1st column)
        gc_rr_resp = calc_gc(df_gc_resp_rr)    
        # Resp->RR (2nd column->1st column)
        gc_resp_rr = calc_gc(df_gc_rr_resp)    

        #STE
        ste_resp_rr,ste_rr_resp = calc_ste(resp_clean, rr)

        #lsNGC
        lsngc_rr_resp, lsngc_resp_rr = calc_lsngc(df_gc_resp_rr)

        # Corr
        corr_coef, corr_lag = calc_corr(df_gc_resp_rr)

        # Mutual information 
        mi = calc_mi(df_gc_resp_rr)

        # Active information https://elife-asu.github.io/PyInform/timeseries.html#module-pyinform.activeinfo
        ai = calc_active_information(df_gc_resp_rr)

        block_en = calc_block_entropy(df_gc_resp_rr)

        cond_en = calc_conditional_entropy(df_gc_resp_rr)

        en_rate = calc_entropy_rate(df_gc_resp_rr)

        trans_en = calc_transfer_entropy(df_gc_resp_rr)

        perm_en = calc_permutation_entropy(df_gc_resp_rr)

        causal_features = pd.DataFrame(
            {
                'ID': file_name,
                'GC_RR_Resp': gc_rr_resp,
                'GC_Resp_RR': gc_resp_rr,
                'STE_Resp_RR':ste_resp_rr,
                'STE_RR_Resp': ste_rr_resp,
                'lsNGC_RR_Resp': lsngc_rr_resp,
                'lsNGC_Resp_RR': lsngc_resp_rr,
                'Corr_coef': corr_coef,
                'Corr_lag': corr_lag,
                'MI': mi,
                'AI': ai,
                'Block_En': block_en,
                'Cond_En': cond_en,
                'En_rate': en_rate,
                'Trans_En': trans_en,
                'Perm_En': perm_en,
            },
            index=[0]
        )

        features = pd.concat([hrv.dropna(axis=1),sym_dyn,resp_features,causal_features],axis=1)
        features = features[['ID'] + [column for column in features.columns if column!='ID']]
        features_all.append(features)

features_all = pd.concat(features_all)

#%%
features_all_2 = features_all[features_all.columns[~np.any(features_all.isnull(),axis=0)]]
# features_all['ID'] = [f'HH{file_id[-7:-4]}' for file_id in features_all['ID']]
features_all_2.to_csv('HR_Resp_causal_features.csv',index=False)
