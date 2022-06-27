#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 10:56:58 2022

@author: justo
"""
import os
import csv
import itertools
import pandas as pd


subs_id = [2,4]
freqs = ['alpha','beta','gamma','delta','theta']
conditions = ['LaS','LaN']


res_cond = []
for condition in conditions:
    res_freq =[]
    for freq in freqs:
        
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
            subs_combined.append(sub_info)
        
        all_nodes = list(itertools.chain(*subs_combined))
        
        
        with open(freq+'_'+condition+'.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(all_nodes)
