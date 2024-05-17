#%%
import pandas as pd
import os
# %%
df_featurmes = pd.DataFrame()
idx = 0
for file in os.listdir('feature_store'):
    if 'csv' in file:
        df = pd.read_csv(f'feature_store/{file}')
        try:
            df = df.rename(columns={'index':'ID'})
            df = df.sort_values('ID')
        except:
            pass
        try:
            df = df.drop('Unnamed: 0', axis=1)
        except:
            pass
        if idx==0:
            df_features = df
        else:
            df_features = pd.merge(df_features,df, on='ID')
        idx+=1

df_features = df_features.fillna(0)
df_features = df_features.loc[:,df_features.nunique()>1]
df_features.to_csv('ALL_FEATURES.csv')
# %%
