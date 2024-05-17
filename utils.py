import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from scipy.stats import skew, kurtosis
from ste import symbolic_TE
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.tsatools import lagmat2ds
from statsmodels.tools.tools import add_constant
from lsngc.utils import lsNGC
from scipy.stats import pearsonr
from sklearn.metrics import mutual_info_score
from pyinform import active_info, block_entropy, conditional_entropy, entropy_rate, transfer_entropy
from ordpy import permutation_entropy

# RESP FEATURES
def calc_resp_features(resp_clean):

    # Find indices of local maxima
    local_max_indices = argrelextrema(resp_clean, np.greater)[0]

    # Find indices of local minima
    local_min_indices = argrelextrema(resp_clean, np.less)[0]

    # instantenous resp rate
    inst_resp_rate = []
    # inspiration time
    insp_time = []
    # expiration time
    exp_time = []
    # inspiration/expiration ratio
    ie_ratio = []
    if local_min_indices[0]<local_max_indices[0]:
        local_extrema = local_min_indices
    else:
        local_extrema = local_max_indices

    for idx, extr_idx in enumerate(local_extrema):
        try:
            if idx>0:
                inst_resp_rate.append(60/((extr_idx-local_extrema[idx-1])/250))
        except:
            pass

    local_extrema_indices = pd.Series(np.sort(np.concatenate([local_min_indices,local_max_indices])))
    insp_exp_time = local_extrema_indices.diff()[1:]/250
    if local_min_indices[0]<local_max_indices[0]:
        insp_time = insp_exp_time[::2]
        exp_time = insp_exp_time[1:][::2]
        for idx in range(min(len(insp_time),len(exp_time))):
            try:
                ie_ratio.append(insp_time.iloc[idx]/exp_time.iloc[idx])
            except:
                pass
    else:
        exp_time = insp_exp_time[::2]
        insp_time = insp_exp_time[1:][::2]
        for idx in range(min(len(insp_time),len(exp_time))):
            try:
                ie_ratio.append(insp_time.iloc[idx]/exp_time.iloc[idx+1])
            except:
                pass

    data_series = pd.Series(ie_ratio)

    # Calculate Q1, Q3 and IQR
    Q1 = data_series.quantile(0.25)
    Q3 = data_series.quantile(0.75)
    IQR = Q3 - Q1

    # Define bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter out outliers from the original list
    ie_ratio = [x for x in ie_ratio if lower_bound <= x <= upper_bound]
    ie_ratio_mean = np.mean(ie_ratio)

    inst_resp_rate_series = pd.Series(inst_resp_rate,index=local_extrema[:-1]) 
    inst_resp_rate_series = inst_resp_rate_series[(inst_resp_rate_series>3)&(inst_resp_rate_series<40)]

    if len(local_min_indices)>len(local_max_indices):
        resp_rate = len(local_min_indices)/((local_min_indices[-1]-local_min_indices[0])/250/60)
    else: 
        resp_rate = len(local_max_indices)/((local_max_indices[-1]-local_max_indices[0])/250/60)

    std_i_resp_rate = np.std(inst_resp_rate_series)
    min_i_resp_rate = np.min(inst_resp_rate_series)
    max_i_resp_rate = np.max(inst_resp_rate_series)
    min_insp_time = np.min(insp_time)
    max_insp_time = np.max(insp_time)
    mean_insp_time = np.mean(insp_time)
    std_insp_time = np.std(insp_time)
    min_exp_time = np.min(exp_time)
    max_exp_time = np.max(exp_time)
    mean_exp_time = np.mean(exp_time)
    std_exp_time = np.std(exp_time)

    inst_tv = []
    for idx, min_idx in enumerate(local_min_indices):
        try:
            if local_min_indices[0]<local_max_indices[0]:
                inst_tv.append(resp_clean[local_max_indices[idx]]-resp_clean[min_idx])
            else:
                inst_tv.append(resp_clean[local_max_indices[idx+1]]-resp_clean[min_idx])
        except:
            pass

    inst_tv = inst_tv/np.median(inst_tv)
    tv_std = np.std(inst_tv)
    tv_q25 = np.quantile(inst_tv,0.25)
    tv_q75 = np.quantile(inst_tv,0.75)
    tv_skew = skew(inst_tv)
    tv_kurtosis = kurtosis(inst_tv)

    resp_features = pd.DataFrame(
        {
            "RespRate": resp_rate,
            "Std_inst_resp_rate": std_i_resp_rate,
            "Min_inst_resp_rate": min_i_resp_rate,
            "Max_inst_resp_rate": max_i_resp_rate,
            "Mean_insp_time": mean_insp_time,
            "Min_insp_time": min_insp_time,
            "Max_insp_time": max_insp_time,
            "Std_insp_time": std_insp_time,
            "Mean_exp_time": mean_exp_time,
            "Min_exp_time": min_exp_time,
            "Max_exp_time": max_exp_time,
            "Std_exp_time": std_exp_time,
            "TV_std": tv_std,
            "TV_q25": tv_q25,
            "TV_q75": tv_q75,
            "TV_skew": tv_skew,
            "TV_kurtosis": tv_kurtosis,
            "IE_ratio_mean": ie_ratio_mean
        },
        index=[0]
    )

    return resp_features

def calc_ste(resp_clean, rr):
    resp_for_symbolic = resp_clean[rr.dropna().index]
    # resp->rr
    ste_resp_rr = symbolic_TE(resp_for_symbolic.values,rr.dropna().values,3,5)
    # rr->resp
    ste_rr_resp = symbolic_TE(rr.dropna().values,resp_for_symbolic.values,3,5)

    return ste_resp_rr,ste_rr_resp

def calc_gc(df_gc):

    maxlag = 25
    granger_result = grangercausalitytests(df_gc, maxlag=[maxlag], verbose=False)

    mdl_1 = granger_result[maxlag][1][0]
    mdl_2 = granger_result[maxlag][1][1]

    dta = lagmat2ds(df_gc, maxlag, trim="both", dropex=1)

    dtaown = add_constant(dta[:, 1 : (maxlag + 1)], prepend=False)
    dtajoint = add_constant(dta[:, 1:], prepend=False)

    y_pred1 = mdl_1.predict(dtaown)
    y_pred2 = mdl_2.predict(dtajoint)
    error1 = dta[:, 0]-y_pred1
    error2 = dta[:, 0]-y_pred2
    gc = np.log(np.var(error1)/np.var(error2))

    return gc

def calc_lsngc(df_gc_resp_rr):
    Aff, f_stat = lsNGC(df_gc_resp_rr.reshape((2,-1)), k_f=2, k_g=2, ar_order=1, normalize=1)
    #rr->resp
    lsngc_rr_resp = Aff[0,1]
    #resp->rr
    lsngc_resp_rr = Aff[1,0]
    
    return lsngc_rr_resp, lsngc_resp_rr

def calc_corr(df_gc_resp_rr: np.array):
    """ returns the highest Pearson correlation coefficient and the lag for which it was obtained between -1 and 1 second

    Args:
        df_gc_resp_rr (np.array): _description_

    Returns:
        _type_: _description_
    """

    corr = []
    for idx in range(-25,0):
        r,_ = pearsonr(df_gc_resp_rr[-idx:,0],df_gc_resp_rr[:idx,1])
        corr.append(r)
    r,_ = pearsonr(df_gc_resp_rr[:,0],df_gc_resp_rr[:,1])
    corr.append(r)
    for idx in range(1,26):
        r,_ = pearsonr(df_gc_resp_rr[idx:,1],df_gc_resp_rr[:-idx,0])
        corr.append(r)
    best_corr = corr[np.argmax(np.abs(corr))]
    best_lag = np.arange(-25,26)[np.argmax(np.abs(corr))]
    return best_corr, best_lag

def calc_mi(df_gc_resp_rr):
    mi = mutual_info_score(df_gc_resp_rr[:,1],df_gc_resp_rr[:,0])
    return mi

def min_max_scaler(df):
    ts1 = df[:,0]-min(df[:,0])
    ts1 = ts1/max(ts1)
    ts2 = df[:,1]-min(df[:,1])
    ts2 = ts2/max(ts2)
    return ts1,ts2

def calc_active_information(df):
    ts1,ts2 = min_max_scaler(df)
    return active_info([ts1,ts2], k=10)

def calc_block_entropy(df):
    ts1,ts2 = min_max_scaler(df)
    return block_entropy([ts1,ts2], k=10)

def calc_conditional_entropy(df):
    ts1,ts2 = min_max_scaler(df)
    return conditional_entropy(ts1,ts2) 

def calc_entropy_rate(df):
    ts1,ts2 = min_max_scaler(df)
    return entropy_rate([ts1,ts2],k=10) 

def calc_transfer_entropy(df):
    ts1,ts2 = min_max_scaler(df)
    return transfer_entropy(ts1,ts2,k=10) 

def calc_permutation_entropy(df):
    ts1,ts2 = min_max_scaler(df)
    return permutation_entropy([ts1,ts2])

methods_dict = {
    'equal_probability_method_4': 'SymDynEqualPorba4',
    'equal_probability_method_6': 'SymDynEqualPorba6',
    'max_min_method': 'SymDynMaxMin',
    'sigma_method': 'SymDynSigma'
}

def max_min_method(rr_intervals, quantization_level=6):
    min_val, max_val = np.min(rr_intervals), np.max(rr_intervals)
    thresholds = np.linspace(min_val, max_val, quantization_level + 1)[1:-1]
    symbols = np.digitize(rr_intervals, thresholds)
    return symbols

def sigma_method(rr_intervals, sigma_rate=0.05, quantization_level=3):
    mu = np.mean(rr_intervals)  # Calculate the mean (μ) of RR intervals
    # Transform RR intervals into symbols based on the given thresholds
    symbols = np.zeros(rr_intervals.shape, dtype=int)

    # Conditions for assigning symbols
    symbols[rr_intervals > (1 + sigma_rate) * mu] = 0
    symbols[(mu < rr_intervals) & (rr_intervals <= (1 + sigma_rate) * mu)] = 1
    symbols[(rr_intervals <= mu) & (rr_intervals > (1 - sigma_rate) * mu)] = 2
    symbols[rr_intervals <= (1 - sigma_rate) * mu] = 3

    return symbols


def equal_probability_method(rr_intervals, quantization_level=4):
    percentiles = np.linspace(0, 100, quantization_level+1)
    # Find the values at those percentiles in the RR interval data
    percentile_values = np.percentile(rr_intervals, percentiles)
    # Digitize the RR intervals according to the percentile values
    # np.digitize bins values into the rightmost bin, so we subtract 1 to correct this
    symbols = np.digitize(rr_intervals, percentile_values, right=False) - 1
    # Ensure all symbols are within the range 0 to quantization_level-1
    symbols[symbols == -1] = 0
    symbols[symbols == quantization_level] = quantization_level - 1
    return symbols

def form_words(symbols):
    words = [symbols[i:i+3] for i in range(len(symbols) - 2)]
    return words

def classify_and_count(words):
    families = {'0V': 0, '1V': 0, '2LV': 0, '2UV': 0}
    for word in words:
        unique_elements = len(set(word))
        if unique_elements == 1:
            families['0V'] += 1
        elif unique_elements == 2:
            families['1V'] += 1
        elif unique_elements == 3:
            if (word[1] > word[0] and word[2] > word[1]) or (word[1] < word[0] and word[2] < word[1]):
                families['2LV'] += 1
            else:
                families['2UV'] += 1

    for key in families.keys():
        families[key] = families[key]/len(words)
    return families

def calc_symdyn(rr_intervals):
    families_all = []
    # Max-min method example
    for method in [max_min_method, sigma_method, equal_probability_method]:
        method_name = method.__name__
        if method_name=='equal_probability_method':
            for quantization_level in [4,6]:
                method_name=method.__name__+f'_{quantization_level}'
                symbols = method(rr_intervals,quantization_level)
                words = form_words(symbols)
                families = classify_and_count(words)
                families = pd.DataFrame(families,index=[0])
                families.columns = [methods_dict[method_name]+f'_{column}' for column in families.columns]
                families_all.append(families)
        else:
            symbols = method(rr_intervals)
            words = form_words(symbols)
            families = classify_and_count(words)
            families = pd.DataFrame(families,index=[0])
            families.columns = [methods_dict[method_name]+f'_{column}' for column in families.columns]
            families_all.append(families)
        
    families_all = pd.concat(families_all,axis=1)
    return families_all