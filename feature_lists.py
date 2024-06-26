demogr_features = ['Wiek','Masa','Wzrost','BMI']

hr_features = [
    'HRV_MeanNN',
    'HRV_SDNN',
    'HRV_SDANN1',
    'HRV_SDNNI1',
    'HRV_RMSSD',
    'HRV_SDSD',
    'HRV_CVNN',
    'HRV_CVSD',
    'HRV_MedianNN',
    'HRV_MadNN',
    'HRV_MCVNN',
    'HRV_IQRNN',
    'HRV_SDRMSSD',
    'HRV_Prc20NN',
    'HRV_Prc80NN',
    'HRV_pNN50',
    'HRV_pNN20',
    'HRV_MinNN',
    'HRV_MaxNN',
    'HRV_HTI',
    'HRV_TINN',
    'HRV_VLF',
    'HRV_LF',
    'HRV_HF',
    'HRV_VHF',
    'HRV_TP',
    'HRV_LFHF',
    'HRV_LFn',
    'HRV_HFn',
    'HRV_LnHF',
    'HRV_SD1',
    'HRV_SD2',
    'HRV_SD1SD2',
    'HRV_S',
    'HRV_CSI',
    'HRV_CVI',
    'HRV_CSI_Modified',
    'HRV_PIP',
    'HRV_IALS',
    'HRV_PSS',
    'HRV_PAS',
    'HRV_GI',
    'HRV_SI',
    'HRV_AI',
    'HRV_PI',
    'HRV_C1d',
    'HRV_C1a',
    'HRV_SD1d',
    'HRV_SD1a',
    'HRV_C2d',
    'HRV_C2a',
    'HRV_SD2d',
    'HRV_SD2a',
    'HRV_Cd',
    'HRV_Ca',
    'HRV_SDNNd',
    'HRV_SDNNa',
    'HRV_DFA_alpha1',
    'HRV_MFDFA_alpha1_Width',
    'HRV_MFDFA_alpha1_Peak',
    'HRV_MFDFA_alpha1_Mean',
    'HRV_MFDFA_alpha1_Max',
    'HRV_MFDFA_alpha1_Delta',
    'HRV_MFDFA_alpha1_Asymmetry',
    'HRV_MFDFA_alpha1_Fluctuation',
    'HRV_MFDFA_alpha1_Increment',
    'HRV_DFA_alpha2',
    'HRV_MFDFA_alpha2_Width',
    'HRV_MFDFA_alpha2_Peak',
    'HRV_MFDFA_alpha2_Mean',
    'HRV_MFDFA_alpha2_Max',
    'HRV_MFDFA_alpha2_Delta',
    'HRV_MFDFA_alpha2_Asymmetry',
    'HRV_MFDFA_alpha2_Fluctuation',
    'HRV_MFDFA_alpha2_Increment',
    'HRV_ApEn',
    'HRV_SampEn',
    'HRV_ShanEn',
    'HRV_FuzzyEn',
    'HRV_MSEn',
    'HRV_CMSEn',
    'HRV_RCMSEn',
    'HRV_CD',
    'HRV_HFD',
    'HRV_KFD',
    'HRV_LZC',
    'SymDynMaxMin_0V',
    'SymDynMaxMin_1V',
    'SymDynMaxMin_2LV',
    'SymDynMaxMin_2UV',
    'SymDynSigma_0V',
    'SymDynSigma_1V',
    'SymDynSigma_2LV',
    'SymDynSigma_2UV',
    'SymDynEqualPorba4_0V',
    'SymDynEqualPorba4_1V',
    'SymDynEqualPorba4_2LV',
    'SymDynEqualPorba4_2UV',
    'SymDynEqualPorba6_0V',
    'SymDynEqualPorba6_1V',
    'SymDynEqualPorba6_2LV',
    'SymDynEqualPorba6_2UV'
]

resp_features = [
    'RespRate',
    'Std_inst_resp_rate',
    'Min_inst_resp_rate',
    'Max_inst_resp_rate',
    'Mean_insp_time',
    'Min_insp_time',
    'Max_insp_time',
    'Std_insp_time',
    'Mean_exp_time',
    'Min_exp_time',
    'Max_exp_time',
    'Std_exp_time',
    'TV_std',
    'TV_q25',
    'TV_q75',
    'TV_skew',
    'TV_kurtosis',
    'IE_ratio_mean'
]