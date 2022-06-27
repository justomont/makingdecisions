#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 10:03:14 2022

@author: justo
"""

import os
import csv
import itertools
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import ks_2samp, normaltest, kruskal
from statsmodels.stats.weightstats import ztest


subs_id = [2,4]
freqs = ['alpha','beta','gamma','delta','theta']
conditions = ['LaS','LaN']

sub_df = pd.DataFrame()

res_freq =[]
for freq in freqs:

    res_cond = []
    for condition in conditions:

        subs_combined = []
        
        for subj in subs_id:
            subj_files = [_ for _ in os.listdir() if (_.endswith(r".csv"))  and '0'+str(subj)+'_' in _]
            RAS_file = [_ for _ in subj_files if  'RAS' in _][0]
            PSD_file = [_ for _ in subj_files if  'Map' in _][0]
            
            ras = df = pd.read_csv(RAS_file)
            psd = df = pd.read_csv(PSD_file).rename(columns = {'Unnamed: 0':'Node'})
            
            sub_info = []
            for index, row in ras.iterrows():
                # order: R A S PSD
                sub_info.append([row['R'],row['A'],row['S'], float(psd[psd['Node']==row['Node']][freq+'_'+condition])])
                sub_df = sub_df.append({'channel':row['Node'], 'psd':float(psd[psd['Node']==row['Node']][freq+'_'+condition]), 'condition':condition, 'fband':freq}, ignore_index=True)
            subs_combined.append(sub_info)
        
        all_nodes = list(itertools.chain(*subs_combined))
        res_cond.append(np.asarray(all_nodes)[:,-1])
    
    res_freq.append(res_cond)
    
    LaS = res_cond[0]
    LaN = res_cond[1]
    
    _, pS = normaltest(LaS)
    _, pN = normaltest(LaN)
    
    print(freq)
    print(pS<0.05) # check normality
    print(pN<0.05)
    
    _, pvalueK = ks_2samp(LaS, LaN) # kolmogorov
    _, pvalueZ = ztest(LaS, LaN) # z-test
    
    print(pvalueK)
    print(pvalueZ)
    
model = ols('psd ~ C(fband, Sum)*C(condition, Sum)', data=sub_df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

H,p = kruskal(sub_df[(sub_df.condition=='LaN') & (sub_df.fband == 'alpha')].psd,   sub_df[(sub_df.condition=='LaS') & (sub_df.fband == 'alpha')].psd)
    
H,p = kruskal(sub_df[sub_df.condition=='LaN'].psd,   sub_df[sub_df.condition=='LaS'].psd)
print('conditions')
print(H,p)

H,p = kruskal(sub_df[sub_df.fband=='delta'].psd,   sub_df[sub_df.fband=='theta'].psd,   sub_df[sub_df.fband=='alpha'].psd,   sub_df[sub_df.fband=='beta'].psd,   sub_df[sub_df.fband=='gamma'].psd)
print('freqs')
print(H,p)


for freq in freqs: 

    Data2 = [sub_df.loc[ids,'psd'].values for ids in sub_df[sub_df.fband == freq].groupby("condition").groups.values()]
    H, p = kruskal(*Data2)
    print('\n'+freq)
    print(H,p)

