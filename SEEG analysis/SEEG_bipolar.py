#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 16:09:02 2022

@author: justo
"""
from IPython import get_ipython
get_ipython().magic('reset -sf')

import os
import mne
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mne.stats import permutation_cluster_1samp_test as pcluster_test

import analysis_combined as analysis

# Load subject
fileDir = r'data_context/'
subj = 4
file_name = [_ for _ in os.listdir(fileDir) if _.endswith(r".EDF") and "P00"+str(subj) in _][0]
sub = analysis.subject(fileDir,subj,tmin ='et_looking_Start', tmax = 'et_decision_Start')
n_trials = len(sub.trials)
LaS_index = sub.trials.index[sub.trials['LaN'] == 1].tolist()
LaN_index = sub.trials.index[sub.trials['LaN'] == 0].tolist()

# Load EDF file
print('Loading EDF+ file...')
raw = mne.io.read_raw_edf(fileDir+file_name,preload=True)

Fs = raw.info['sfreq']

# Check the raw signal
raw.plot_psd()
raw.plot()

# Notch filter
raw.notch_filter(np.arange(50, 251, 50),phase='zero-double')

# Low pass filter to remove any possible low drift 
raw.filter(l_freq=1.,h_freq=None,phase='zero-double')

#%%
# Set channel types
print('Setting channel types...')
channel_types = {}
channels = raw.info.ch_names
for channel in range(len(channels)):
    if channels[channel] == 'TTL':
        channel_types[channels[channel]] = 'stim' 
    elif 'ECG' in channels[channel]: 
        channel_types[channels[channel]] = 'ecg'
    else:
        channel_types[channels[channel]] = 'seeg'
raw.set_channel_types(channel_types)

# Check the filtered signal
raw.plot_psd(picks=['seeg'])
raw.plot()

#%%Bipolar montage

single_els = []
exclude = ['TTL','ECG']
for name in raw.ch_names: 
    if ''.join([i for i in name if not i.isdigit()]) not in single_els:
        if ''.join([i for i in name if not i.isdigit()]) not in exclude:
            single_els.append(''.join([i for i in name if not i.isdigit()]))

max_n_els = []
for letter in single_els:
    count = []
    for name in raw.ch_names: 
        if letter == ''.join([i for i in name if not i.isdigit()]):
            count.append(name)
    max_n_els.append(count)
    

raw_bip_ref = raw.copy()
for electrode in max_n_els:
    raw_bip_ref = mne.set_bipolar_reference(raw_bip_ref, anode=electrode[:-1],cathode=electrode[1:])

raw = raw_bip_ref
raw.plot_psd(picks=['seeg'])
raw.plot()
#%%
# ICA to remove noise from signal
ica_question = input('Use ICA? [Y/N]: ')
if ica_question == 'Y':
    print('Performing ICA...')
    ica_components = int(input('Number of components for the ICA:'))
    ica = mne.preprocessing.ICA(n_components=ica_components,method='picard')
    ica.fit(raw)
    
    # Plot ICs
    ica.plot_sources(raw, show_scrollbars=False)
    
    # Copy and reconstruct the signal exluding the ICs
    ica.exclude = np.arange(ica_components)
    reconst_raw = raw.copy()
    ica.apply(reconst_raw)
    
    # Compare raw and ICA filtered signal 
    raw.plot()
    reconst_raw.plot()
else:
    reconst_raw = raw.copy()

#%% Obtain events from annotations WHEN THE TTL IS NOT WORKING
_, event_id = mne.events_from_annotations(raw)

# Change events id so that: 0 for LaN, 1 for LaS
LaS_event_id = {key: 1 for key in event_id if 'TRIAL' in key and int(key.split()[-1]) in LaS_index}
LaN_event_id = {key: 0 for key in event_id if 'TRIAL' in key and int(key.split()[-1]) in LaN_index}
event_dict = LaS_event_id | LaN_event_id
events, _ = mne.events_from_annotations(raw, event_id=event_dict)
event_id = dict(LaN=0,LaS=1)

epochs = mne.Epochs(reconst_raw, events, event_id, baseline=None, detrend=1, tmax=10, preload=True)


#%% Time-frequency

# freqs = np.logspace(*np.log10([1, 100]), num=30)
# n_cycles = freqs/2.  # different number of cycle per frequency
# power, itc = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True,return_itc=True,decim=3, average=True)
# baseline = [4,5]


#%% Channels of interes

# all channels names
all_channels = epochs.pick_types(seeg=True) 
# select just the single electrode names (not the specific contact)
single_electrodes = np.unique([ ''.join((x for x in name if not x.isdigit())) for name in all_channels.ch_names if ''.join((x for x in name if not x.isdigit())) != 'C']).tolist()
#  list of lists for each electrode and its contacts
all_electrodes = [[name for name in all_channels.ch_names if electrode == ''.join((x for x in name if not x.isdigit()))] for electrode in single_electrodes]

# %% Select relevant nodes for the sub 

input('Remember to add the variable for the subject! [Press Enter]')
if subj==4:
    subject_nodes = [['Node', 'R', 'A', 'S', 'i', 'j', 'k', 'Anatomical Label', 'Anatomic Region'], ["F'1", -26.34000015258789, 31.809999465942383, 29.020000457763672, 154, 98, 159, 2, 'Left-Cerebral-White-Matter'], ["F'2", -29.793854658271997, 31.77436820689169, 29.585349435779957, 157, 98, 159, 2, 'Left-Cerebral-White-Matter'], ["F'3", -33.2477091639561, 31.738736947840998, 30.150698413796245, 161, 97, 159, 3, 'Left-Cerebral-Cortex'], ["F'4", -36.70156366964021, 31.703105688790306, 30.71604739181253, 164, 97, 159, 3, 'Left-Cerebral-Cortex'], ["F'5", -40.155418175324314, 31.667474429739613, 31.28139636982882, 168, 96, 159, 3, 'Left-Cerebral-Cortex'], ["F'6", -43.60927268100842, 31.631843170688917, 31.846745347845104, 171, 96, 159, 2, 'Left-Cerebral-White-Matter'], ["F'7", -47.063127186692526, 31.596211911638225, 32.41209432586139, 175, 95, 159, 2, 'Left-Cerebral-White-Matter'], ["F'8", -50.51698169237663, 31.560580652587532, 32.97744330387768, 178, 95, 159, 3, 'Left-Cerebral-Cortex'], ["HA'1", -20.43000030517578, 11.84000015258789, -8.270000457763672, 148, 136, 139, 17, 'Left-Hippocampus'], ["HA'2", -23.91220701212441, 11.760454825452118, -7.926621671271449, 151, 135, 139, 17, 'Left-Hippocampus'], ["HA'3", -27.394413719073032, 11.680909498316346, -7.583242884779225, 155, 135, 139, 17, 'Left-Hippocampus'], ["HA'4", -30.87662042602166, 11.601364171180574, -7.239864098287002, 158, 135, 139, 5, 'Left-Inf-Lat-Vent'], ["HA'5", -34.35882713297028, 11.521818844044802, -6.8964853117947795, 162, 134, 139, 2, 'Left-Cerebral-White-Matter'], ["HA'6", -37.84103383991891, 11.442273516909031, -6.553106525302557, 165, 134, 139, 2, 'Left-Cerebral-White-Matter'], ["HA'7", -41.32324054686754, 11.362728189773259, -6.209727738810333, 169, 134, 139, 2, 'Left-Cerebral-White-Matter'], ["HA'8", -44.80544725381617, 11.283182862637487, -5.86634895231811, 172, 133, 139, 2, 'Left-Cerebral-White-Matter'], ["HA'9", -48.287653960764786, 11.203637535501715, -5.522970165825887, 176, 133, 139, 3, 'Left-Cerebral-Cortex'], ["HA'10", -51.76986066771342, 11.124092208365942, -5.179591379333663, 179, 133, 139, 3, 'Left-Cerebral-Cortex'], ["HA'11", -55.25206737466204, 11.04454688123017, -4.836212592841441, 183, 132, 139, 3, 'Left-Cerebral-Cortex'], ["HA'12", -58.73427408161067, 10.965001554094398, -4.492833806349218, 186, 132, 138, 3, 'Left-Cerebral-Cortex'], ["HP'1", -11.501999855041504, -16.342750549316406, 5.6479997634887695, 139, 122, 111, 17, 'Left-Hippocampus'], ["HP'2", -14.894704503345976, -17.148658788016107, 5.347887984031215, 142, 122, 110, 17, 'Left-Hippocampus'], ["HP'3", -18.287409151650447, -17.95456702671581, 5.04777620457366, 146, 122, 110, 17, 'Left-Hippocampus'], ["HP'4", -21.680113799954917, -18.760475265415515, 4.747664425116105, 149, 123, 109, 2, 'Left-Cerebral-White-Matter'], ["HP'5", -25.07281844825939, -19.566383504115215, 4.447552645658551, 153, 123, 108, 2, 'Left-Cerebral-White-Matter'], ["HP'6", -28.46552309656386, -20.372291742814916, 4.147440866200997, 156, 123, 107, 2, 'Left-Cerebral-White-Matter'], ["HP'7", -31.858227744868334, -21.17819998151462, 3.847329086743441, 159, 124, 106, 2, 'Left-Cerebral-White-Matter'], ["HP'8", -35.2509323931728, -21.984108220214324, 3.547217307285887, 163, 124, 106, 2, 'Left-Cerebral-White-Matter'], ["HP'9", -38.64363704147728, -22.790016458914025, 3.247105527828332, 166, 124, 105, 2, 'Left-Cerebral-White-Matter'], ["HP'10", -42.03634168978175, -23.595924697613725, 2.9469937483707773, 170, 125, 104, 2, 'Left-Cerebral-White-Matter'], ["HP'11", -45.429046338086216, -24.40183293631343, 2.646881968913223, 173, 125, 103, 2, 'Left-Cerebral-White-Matter'], ["HP'12", -48.82175098639069, -25.207741175013133, 2.3467701894556683, 176, 125, 102, 2, 'Left-Cerebral-White-Matter'], ["HP'13", -52.21445563469516, -26.013649413712834, 2.046658409998113, 180, 125, 101, 3, 'Left-Cerebral-Cortex'], ["HP'14", -55.60716028299963, -26.819557652412534, 1.7465466305405588, 183, 126, 101, 0, 'Unknown'], ["HP'15", -58.9998649313041, -27.62546589111224, 1.446434851083004, 186, 126, 100, 0, 'Unknown'], ["IC'1", -34.12595748901367, 19.8756046295166, 0.6579999923706055, 162, 127, 147, 2, 'Left-Cerebral-White-Matter'], ["IC'2", -33.75455448355482, 18.645327513840027, 3.913530367928275, 161, 124, 146, 2, 'Left-Cerebral-White-Matter'], ["IC'3", -33.38315147809597, 17.41505039816345, 7.169060743485945, 161, 120, 145, 2, 'Left-Cerebral-White-Matter'], ["IC'4", -33.011748472637116, 16.184773282486876, 10.424591119043614, 161, 117, 144, 2, 'Left-Cerebral-White-Matter'], ["IC'5", -32.64034546717827, 14.954496166810301, 13.680121494601284, 160, 114, 142, 2, 'Left-Cerebral-White-Matter'], ["IC'6", -32.26894246171942, 13.724219051133726, 16.935651870158956, 160, 111, 141, 3, 'Left-Cerebral-Cortex'], ["IC'7", -31.897539456260567, 12.493941935457151, 20.19118224571662, 159, 107, 140, 2, 'Left-Cerebral-White-Matter'], ["IC'8", -31.526136450801715, 11.263664819780576, 23.44671262127429, 159, 104, 139, 3, 'Left-Cerebral-Cortex'], ["IC'9", -31.154733445342863, 10.033387704104, 26.702242996831963, 159, 101, 138, 3, 'Left-Cerebral-Cortex'], ["IC'10", -30.78333043988401, 8.803110588427424, 29.957773372389635, 158, 98, 136, 2, 'Left-Cerebral-White-Matter'], ["IC'11", -30.411927434425163, 7.572833472750849, 33.21330374794731, 158, 94, 135, 2, 'Left-Cerebral-White-Matter'], ["IC'12", -30.04052442896631, 6.342556357074274, 36.46883412350497, 158, 91, 134, 2, 'Left-Cerebral-White-Matter'], ["IC'13", -29.66912142350746, 5.112279241397699, 39.72436449906264, 157, 88, 133, 2, 'Left-Cerebral-White-Matter'], ["IC'14", -29.29771841804861, 3.882002125721124, 42.979894874620314, 157, 85, 131, 2, 'Left-Cerebral-White-Matter'], ["IC'15", -28.92631541258976, 2.651725010044551, 46.235425250177975, 156, 81, 130, 2, 'Left-Cerebral-White-Matter'], ["IC'16", -28.554912407130907, 1.4214478943679758, 49.49095562573565, 156, 78, 129, 2, 'Left-Cerebral-White-Matter'], ["IC'17", -28.183509401672055, 0.1911707786913972, 52.74648600129332, 156, 75, 128, 2, 'Left-Cerebral-White-Matter'], ["IC'18", -27.812106396213206, -1.0391063369851743, 56.00201637685098, 155, 71, 126, 2, 'Left-Cerebral-White-Matter'], ["IF'1", -35.021915435791016, 26.796520233154297, 6.6479997634887695, 163, 121, 154, 3, 'Left-Cerebral-Cortex'], ["IF'2", -34.15873417988489, 26.280817346567552, 10.000456870182072, 162, 117, 154, 3, 'Left-Cerebral-Cortex'], ["IF'3", -33.29555292397877, 25.765114459980808, 13.352913976875374, 161, 114, 153, 2, 'Left-Cerebral-White-Matter'], ["IF'4", -32.43237166807264, 25.249411573394063, 16.70537108356868, 160, 111, 153, 3, 'Left-Cerebral-Cortex'], ["IF'5", -31.56919041216652, 24.73370868680732, 20.05782819026198, 159, 107, 152, 2, 'Left-Cerebral-White-Matter'], ["IF'6", -30.706009156260397, 24.218005800220574, 23.41028529695528, 158, 104, 152, 3, 'Left-Cerebral-Cortex'], ["IF'7", -29.842827900354273, 23.702302913633826, 26.762742403648584, 157, 101, 151, 2, 'Left-Cerebral-White-Matter'], ["IF'8", -28.979646644448145, 23.18660002704708, 30.115199510341885, 156, 97, 151, 2, 'Left-Cerebral-White-Matter'], ["IF'9", -28.116465388542025, 22.670897140460337, 33.46765661703519, 156, 94, 150, 2, 'Left-Cerebral-White-Matter'], ["IF'10", -27.253284132635898, 22.155194253873592, 36.82011372372849, 155, 91, 150, 2, 'Left-Cerebral-White-Matter'], ["IF'11", -26.390102876729777, 21.639491367286848, 40.17257083042179, 154, 87, 149, 2, 'Left-Cerebral-White-Matter'], ["IF'12", -25.52692162082365, 21.123788480700103, 43.52502793711509, 153, 84, 149, 2, 'Left-Cerebral-White-Matter'], ["IF'13", -24.663740364917526, 20.608085594113355, 46.8774850438084, 152, 81, 148, 2, 'Left-Cerebral-White-Matter'], ["IF'14", -23.800559109011402, 20.09238270752661, 50.22994215050169, 151, 77, 148, 2, 'Left-Cerebral-White-Matter'], ["IF'15", -22.93737785310528, 19.576679820939866, 53.582399257195, 150, 74, 147, 2, 'Left-Cerebral-White-Matter'], ["IF'16", -22.074196597199155, 19.06097693435312, 56.934856363888294, 150, 71, 147, 3, 'Left-Cerebral-Cortex'], ["IF'17", -21.21101534129303, 18.545274047766377, 60.2873134705816, 149, 67, 146, 0, 'Unknown'], ["IF'18", -20.347834085386907, 18.029571161179632, 63.639770577274895, 148, 64, 146, 3, 'Left-Cerebral-Cortex'], ["IP'1", -31.8629150390625, 8.495521545410156, 5.690999984741211, 159, 122, 136, 2, 'Left-Cerebral-White-Matter'], ["IP'2", -31.54365927693893, 6.74974649309963, 8.707677830498207, 159, 119, 134, 2, 'Left-Cerebral-White-Matter'], ["IP'3", -31.224403514815354, 5.003971440789103, 11.724355676255204, 159, 116, 133, 2, 'Left-Cerebral-White-Matter'], ["IP'4", -30.905147752691782, 3.2581963884785754, 14.741033522012202, 158, 113, 131, 2, 'Left-Cerebral-White-Matter'], ["IP'5", -30.585891990568207, 1.5124213361680487, 17.757711367769197, 158, 110, 129, 3, 'Left-Cerebral-Cortex'], ["IP'6", -30.266636228444636, -0.2333537161424779, 20.774389213526195, 158, 107, 127, 3, 'Left-Cerebral-Cortex'], ["IP'7", -29.947380466321064, -1.9791287684530054, 23.791067059283193, 157, 104, 126, 3, 'Left-Cerebral-Cortex'], ["IP'8", -29.62812470419749, -3.724903820763531, 26.807744905040188, 157, 101, 124, 3, 'Left-Cerebral-Cortex'], ["IP'9", -29.308868942073918, -5.470678873074059, 29.824422750797186, 157, 98, 122, 2, 'Left-Cerebral-White-Matter'], ["IP'10", -28.989613179950343, -7.216453925384586, 32.841100596554185, 156, 95, 120, 2, 'Left-Cerebral-White-Matter'], ["IP'11", -28.67035741782677, -8.962228977695112, 35.85777844231118, 156, 92, 119, 2, 'Left-Cerebral-White-Matter'], ["IP'12", -28.351101655703197, -10.70800403000564, 38.874456288068174, 156, 89, 117, 2, 'Left-Cerebral-White-Matter'], ["IP'13", -28.031845893579625, -12.453779082316167, 41.891134133825176, 156, 86, 115, 2, 'Left-Cerebral-White-Matter'], ["IP'14", -27.712590131456054, -14.199554134626695, 44.90781197958217, 155, 83, 113, 2, 'Left-Cerebral-White-Matter'], ["IP'15", -27.39333436933248, -15.945329186937219, 47.924489825339165, 155, 80, 112, 2, 'Left-Cerebral-White-Matter'], ["IP'16", -27.074078607208907, -17.691104239247746, 50.94116767109617, 155, 77, 110, 2, 'Left-Cerebral-White-Matter'], ["IP'17", -26.754822845085336, -19.436879291558274, 53.95784551685316, 154, 74, 108, 2, 'Left-Cerebral-White-Matter'], ["IP'18", -26.43556708296176, -21.1826543438688, 56.97452336261016, 154, 71, 106, 2, 'Left-Cerebral-White-Matter'], ["O'1", -2.568915605545044, -56.72035217285156, -2.0980000495910645, 130, 130, 71, 2, 'Left-Cerebral-White-Matter'], ["O'2", -6.053530659759835, -56.63181620055919, -1.7823747883240082, 134, 129, 71, 2, 'Left-Cerebral-White-Matter'], ["O'3", -9.538145713974625, -56.54328022826682, -1.466749527056952, 137, 129, 71, 2, 'Left-Cerebral-White-Matter'], ["O'4", -13.022760768189416, -56.454744255974454, -1.1511242657898957, 141, 129, 71, 2, 'Left-Cerebral-White-Matter'], ["O'5", -16.507375822404207, -56.366208283682084, -0.8354990045228397, 144, 128, 71, 2, 'Left-Cerebral-White-Matter'], ["O'6", -19.991990876618996, -56.277672311389715, -0.5198737432557834, 147, 128, 71, 2, 'Left-Cerebral-White-Matter'], ["O'7", -23.476605930833788, -56.189136339097345, -0.20424848198872692, 151, 128, 71, 3, 'Left-Cerebral-Cortex'], ["O'8", -26.961220985048577, -56.100600366804976, 0.11137677927832934, 154, 127, 71, 3, 'Left-Cerebral-Cortex'], ["O'9", -30.445836039263366, -56.012064394512606, 0.42700204054538515, 158, 127, 71, 2, 'Left-Cerebral-White-Matter'], ["O'10", -33.930451093478155, -55.92352842222024, 0.7426273018124414, 161, 127, 72, 2, 'Left-Cerebral-White-Matter'], ["O'11", -37.41506614769295, -55.83499244992787, 1.0582525630794977, 165, 126, 72, 3, 'Left-Cerebral-Cortex'], ["O'12", -40.89968120190774, -55.7464564776355, 1.373877824346554, 168, 126, 72, 3, 'Left-Cerebral-Cortex'], ["OP'1", -32.258914947509766, 1.0616055727005005, 30.077999114990234, 160, 97, 129, 2, 'Left-Cerebral-White-Matter'], ["OP'2", -35.700599761614924, 1.1239480648265947, 29.444817423819817, 163, 98, 129, 2, 'Left-Cerebral-White-Matter'], ["OP'3", -39.14228457572009, 1.186290556952689, 28.811635732649403, 167, 99, 129, 3, 'Left-Cerebral-Cortex'], ["OP'4", -42.58396938982525, 1.2486330490787831, 28.178454041478986, 170, 99, 129, 2, 'Left-Cerebral-White-Matter'], ["OP'5", -46.02565420393041, 1.3109755412048774, 27.54527235030857, 174, 100, 129, 2, 'Left-Cerebral-White-Matter'], ["OP'6", -49.467339018035574, 1.3733180333309716, 26.912090659138155, 177, 101, 129, 2, 'Left-Cerebral-White-Matter'], ["OP'7", -52.90902383214073, 1.4356605254570656, 26.278908967967737, 180, 101, 129, 2, 'Left-Cerebral-White-Matter'], ["OP'8", -56.35070864624589, 1.4980030175831598, 25.64572727679732, 184, 102, 129, 3, 'Left-Cerebral-Cortex'], ["OS'1", -2.3959999084472656, -52.637351989746094, 8.406000137329102, 130, 119, 75, 2, 'Left-Cerebral-White-Matter'], ["OS'2", -5.784128700311376, -53.044333614792606, 9.183784970439095, 133, 118, 74, 3, 'Left-Cerebral-Cortex'], ["OS'3", -9.172257492175486, -53.451315239839126, 9.961569803549091, 137, 118, 74, 2, 'Left-Cerebral-White-Matter'], ["OS'4", -12.560386284039597, -53.85829686488564, 10.739354636659087, 140, 117, 74, 2, 'Left-Cerebral-White-Matter'], ["OS'5", -15.948515075903707, -54.26527848993215, 11.51713946976908, 143, 116, 73, 3, 'Left-Cerebral-Cortex'], ["OS'6", -19.336643867767815, -54.67226011497866, 12.294924302879075, 147, 115, 73, 3, 'Left-Cerebral-Cortex'], ["OS'7", -22.724772659631927, -55.07924174002518, 13.07270913598907, 150, 114, 72, 0, 'Unknown'], ["OS'8", -26.112901451496036, -55.486223365071695, 13.850493969099066, 154, 114, 72, 0, 'Unknown'], ["OS'9", -29.501030243360148, -55.89320499011821, 14.62827880220906, 157, 113, 72, 2, 'Left-Cerebral-White-Matter'], ["OS'10", -32.88915903522425, -56.30018661516472, 15.406063635319054, 160, 112, 71, 3, 'Left-Cerebral-Cortex'], ["OS'11", -36.277287827088365, -56.70716824021124, 16.183848468429048, 164, 111, 71, 3, 'Left-Cerebral-Cortex'], ["OS'12", -39.66541661895248, -57.11414986525775, 16.961633301539045, 167, 111, 70, 0, 'Unknown'], ["P'1", -1.9479577541351318, -22.994394302368164, 28.400999069213867, 129, 99, 105, 3, 'Left-Cerebral-Cortex'], ["P'2", -5.400780842345379, -23.10286660227958, 28.96335703499205, 133, 99, 104, 2, 'Left-Cerebral-White-Matter'], ["P'3", -8.853603930555627, -23.211338902191, 29.525715000770234, 136, 98, 104, 2, 'Left-Cerebral-White-Matter'], ["P'4", -12.306427018765875, -23.319811202102414, 30.088072966548417, 140, 97, 104, 2, 'Left-Cerebral-White-Matter'], ["P'5", -15.759250106976122, -23.42828350201383, 30.650430932326596, 143, 97, 104, 2, 'Left-Cerebral-White-Matter'], ["P'6", -19.21207319518637, -23.53675580192525, 31.21278889810478, 147, 96, 104, 2, 'Left-Cerebral-White-Matter'], ["P'7", -22.66489628339662, -23.645228101836665, 31.775146863882963, 150, 96, 104, 2, 'Left-Cerebral-White-Matter'], ["P'8", -26.117719371606867, -23.75370040174808, 32.33750482966114, 154, 95, 104, 2, 'Left-Cerebral-White-Matter'], ["P'9", -29.570542459817112, -23.862172701659496, 32.899862795439326, 157, 95, 104, 2, 'Left-Cerebral-White-Matter'], ["P'10", -33.02336554802736, -23.970645001570915, 33.46222076121751, 161, 94, 104, 2, 'Left-Cerebral-White-Matter'], ["P'11", -36.47618863623761, -24.07911730148233, 34.02457872699569, 164, 93, 103, 2, 'Left-Cerebral-White-Matter'], ["P'12", -39.92901172444786, -24.187589601393746, 34.586936692773875, 167, 93, 103, 2, 'Left-Cerebral-White-Matter'], ["P'13", -43.381834812658106, -24.296061901305166, 35.14929465855206, 171, 92, 103, 3, 'Left-Cerebral-Cortex'], ["P'14", -46.83465790086836, -24.40453420121658, 35.71165262433024, 174, 92, 103, 2, 'Left-Cerebral-White-Matter'], ["P'15", -50.2874809890786, -24.513006501127997, 36.274010590108425, 178, 91, 103, 3, 'Left-Cerebral-Cortex'], ["PI'1", -2.364000082015991, -14.061521530151367, 32.53099822998047, 130, 95, 113, 2, 'Left-Cerebral-White-Matter'], ["PI'2", -5.856584641289487, -13.84027047106404, 32.47713669289606, 133, 95, 114, 2, 'Left-Cerebral-White-Matter'], ["PI'3", -9.349169200562983, -13.619019411976714, 32.42327515581165, 137, 95, 114, 2, 'Left-Cerebral-White-Matter'], ["PI'4", -12.84175375983648, -13.397768352889386, 32.36941361872724, 140, 95, 114, 2, 'Left-Cerebral-White-Matter'], ["PI'5", -16.334338319109975, -13.17651729380206, 32.315552081642835, 144, 95, 114, 2, 'Left-Cerebral-White-Matter'], ["PI'6", -19.826922878383474, -12.955266234714733, 32.26169054455843, 147, 95, 115, 2, 'Left-Cerebral-White-Matter'], ["PI'7", -23.31950743765697, -12.734015175627407, 32.20782900747402, 151, 95, 115, 2, 'Left-Cerebral-White-Matter'], ["PI'8", -26.81209199693047, -12.51276411654008, 32.15396747038961, 154, 95, 115, 2, 'Left-Cerebral-White-Matter'], ["PI'9", -30.30467655620396, -12.291513057452754, 32.1001059333052, 158, 95, 115, 2, 'Left-Cerebral-White-Matter'], ["PI'10", -33.797261115477454, -12.070261998365426, 32.04624439622079, 161, 95, 115, 2, 'Left-Cerebral-White-Matter'], ["PI'11", -37.28984567475096, -11.849010939278099, 31.992382859136384, 165, 96, 116, 2, 'Left-Cerebral-White-Matter'], ["PI'12", -40.78243023402445, -11.627759880190773, 31.938521322051976, 168, 96, 116, 2, 'Left-Cerebral-White-Matter'], ["PI'13", -44.27501479329795, -11.406508821103447, 31.884659784967567, 172, 96, 116, 2, 'Left-Cerebral-White-Matter'], ["PI'14", -47.76759935257144, -11.18525776201612, 31.830798247883155, 175, 96, 116, 2, 'Left-Cerebral-White-Matter'], ["PI'15", -51.26018391184495, -10.964006702928792, 31.776936710798747, 179, 96, 117, 3, 'Left-Cerebral-Cortex'], ["PI'16", -54.752768471118436, -10.742755643841466, 31.72307517371434, 182, 96, 117, 3, 'Left-Cerebral-Cortex'], ["PI'17", -58.24535303039193, -10.521504584754139, 31.66921363662993, 186, 96, 117, 0, 'Unknown'], ["PI'18", -61.73793758966543, -10.300253525666813, 31.61535209954552, 189, 96, 117, 0, 'Unknown'], ["S'1", -1.340042233467102, -8.767605781555176, 33.06100082397461, 129, 94, 119, 3, 'Left-Cerebral-Cortex'], ["S'2", -4.796037883124191, -8.484023728658286, 33.53605375663165, 132, 94, 119, 2, 'Left-Cerebral-White-Matter'], ["S'3", -8.25203353278128, -8.200441675761395, 34.011106689288695, 136, 93, 119, 2, 'Left-Cerebral-White-Matter'], ["S'4", -11.70802918243837, -7.916859622864505, 34.48615962194574, 139, 93, 120, 2, 'Left-Cerebral-White-Matter'], ["S'5", -15.164024832095459, -7.633277569967615, 34.96121255460278, 143, 93, 120, 2, 'Left-Cerebral-White-Matter'], ["S'6", -18.620020481752547, -7.3496955170707245, 35.436265487259824, 146, 92, 120, 2, 'Left-Cerebral-White-Matter'], ["S'7", -22.076016131409638, -7.066113464173835, 35.91131841991687, 150, 92, 120, 2, 'Left-Cerebral-White-Matter'], ["S'8", -25.532011781066725, -6.782531411276945, 36.3863713525739, 153, 91, 121, 2, 'Left-Cerebral-White-Matter'], ["S'9", -28.988007430723815, -6.498949358380054, 36.86142428523095, 156, 91, 121, 2, 'Left-Cerebral-White-Matter'], ["S'10", -32.44400308038091, -6.215367305483165, 37.33647721788799, 160, 90, 121, 2, 'Left-Cerebral-White-Matter'], ["S'11", -35.89999873003799, -5.931785252586274, 37.81153015054503, 163, 90, 122, 2, 'Left-Cerebral-White-Matter'], ["S'12", -39.35599437969508, -5.648203199689384, 38.286583083202075, 167, 89, 122, 2, 'Left-Cerebral-White-Matter'], ["S'13", -42.811990029352174, -5.364621146792493, 38.76163601585912, 170, 89, 122, 2, 'Left-Cerebral-White-Matter'], ["S'14", -46.267985679009264, -5.081039093895603, 39.23668894851616, 174, 88, 122, 2, 'Left-Cerebral-White-Matter'], ["S'15", -49.72398132866635, -4.797457040998713, 39.711741881173204, 177, 88, 123, 2, 'Left-Cerebral-White-Matter'], ["TI'1", -38.95195770263672, 10.559563636779785, 6.364999771118164, 166, 121, 138, 3, 'Left-Cerebral-Cortex'], ["TI'2", -42.3182835704693, 9.627714315101686, 6.587501441335041, 170, 121, 137, 0, 'Unknown'], ["TI'3", -45.68460943830188, 8.695864993423587, 6.810003111551918, 173, 121, 136, 3, 'Left-Cerebral-Cortex'], ["TI'4", -49.050935306134456, 7.7640156717454865, 7.032504781768796, 177, 120, 135, 2, 'Left-Cerebral-White-Matter'], ["TI'5", -52.417261173967034, 6.8321663500673875, 7.255006451985673, 180, 120, 134, 2, 'Left-Cerebral-White-Matter'], ["TI'6", -55.78358704179961, 5.9003170283892885, 7.47750812220255, 183, 120, 133, 3, 'Left-Cerebral-Cortex'], ["TI'7", -59.14991290963219, 4.968467706711189, 7.7000097924194275, 187, 120, 132, 0, 'Unknown'], ["TI'8", -62.51623877746477, 4.036618385033089, 7.9225114626363045, 190, 120, 132, 0, 'Unknown'], ["TP'1", -8.308084487915039, -40.13751983642578, 1.7200000286102295, 136, 126, 87, 2, 'Left-Cerebral-White-Matter'], ["TP'2", -11.803183956102252, -40.322631653277455, 1.7236500613940826, 139, 126, 87, 2, 'Left-Cerebral-White-Matter'], ["TP'3", -15.298283424289467, -40.50774347012912, 1.7273000941779355, 143, 126, 87, 2, 'Left-Cerebral-White-Matter'], ["TP'4", -18.793382892476682, -40.692855286980794, 1.7309501269617886, 146, 126, 87, 3, 'Left-Cerebral-Cortex'], ["TP'5", -22.288482360663895, -40.87796710383247, 1.7346001597456415, 150, 126, 87, 2, 'Left-Cerebral-White-Matter'], ["TP'6", -25.78358182885111, -41.063078920684134, 1.7382501925294946, 153, 126, 86, 2, 'Left-Cerebral-White-Matter'], ["TP'7", -29.278681297038325, -41.24819073753581, 1.7419002253133475, 157, 126, 86, 2, 'Left-Cerebral-White-Matter'], ["TP'8", -32.77378076522554, -41.433302554387474, 1.7455502580972007, 160, 126, 86, 2, 'Left-Cerebral-White-Matter'], ["TP'9", -36.26888023341275, -41.61841437123915, 1.7492002908810536, 164, 126, 86, 3, 'Left-Cerebral-Cortex'], ["TP'10", -39.763979701599965, -41.80352618809082, 1.7528503236649067, 167, 126, 86, 3, 'Left-Cerebral-Cortex'], ["TP'11", -43.25907916978718, -41.98863800494249, 1.7565003564487598, 171, 126, 86, 3, 'Left-Cerebral-Cortex'], ["TP'12", -46.7541786379744, -42.17374982179416, 1.7601503892326127, 174, 126, 85, 3, 'Left-Cerebral-Cortex'], ["TP'13", -50.24927810616161, -42.358861638645834, 1.7638004220164658, 178, 126, 85, 0, 'Unknown'], ["TP'14", -53.744377574348825, -42.5439734554975, 1.7674504548003187, 181, 126, 85, 0, 'Unknown'], ["TP'15", -57.23947704253604, -42.729085272349174, 1.7711004875841718, 185, 126, 85, 0, 'Unknown'], ["TS'1", -33.63399887084961, -9.140563011169434, 13.685999870300293, 161, 114, 118, 2, 'Left-Cerebral-White-Matter'], ["TS'2", -37.05630955691996, -9.872144946045537, 13.736768370097032, 165, 114, 118, 2, 'Left-Cerebral-White-Matter'], ["TS'3", -40.47862024299031, -10.603726880921641, 13.78753686989377, 168, 114, 117, 2, 'Left-Cerebral-White-Matter'], ["TS'4", -43.900930929060664, -11.335308815797745, 13.83830536969051, 171, 114, 116, 2, 'Left-Cerebral-White-Matter'], ["TS'5", -47.32324161513102, -12.06689075067385, 13.889073869487248, 175, 114, 115, 2, 'Left-Cerebral-White-Matter'], ["TS'6", -50.74555230120137, -12.798472685549953, 13.939842369283987, 178, 114, 115, 2, 'Left-Cerebral-White-Matter'], ["TS'7", -54.16786298727172, -13.530054620426055, 13.990610869080726, 182, 114, 114, 3, 'Left-Cerebral-Cortex'], ["TS'8", -57.59017367334207, -14.26163655530216, 14.041379368877465, 185, 113, 113, 3, 'Left-Cerebral-Cortex'], ["TS'9", -61.012484359412426, -14.993218490178263, 14.092147868674203, 189, 113, 113, 0, 'Unknown'], ["TS'10", -64.43479504548277, -15.724800425054365, 14.142916368470942, 192, 113, 112, 0, 'Unknown'], ["W'1", -2.0840423107147217, -29.120479583740234, 21.812999725341797, 130, 106, 98, 2, 'Left-Cerebral-White-Matter'], ["W'2", -5.57869156013353, -29.294871791398858, 21.72925020925657, 133, 106, 98, 2, 'Left-Cerebral-White-Matter'], ["W'3", -9.073340809552338, -29.469263999057482, 21.645500693171343, 137, 106, 98, 2, 'Left-Cerebral-White-Matter'], ["W'4", -12.567990058971146, -29.643656206716102, 21.56175117708612, 140, 106, 98, 2, 'Left-Cerebral-White-Matter'], ["W'5", -16.062639308389954, -29.818048414374726, 21.478001661000892, 144, 106, 98, 2, 'Left-Cerebral-White-Matter'], ["W'6", -19.557288557808764, -29.99244062203335, 21.394252144915665, 147, 106, 98, 2, 'Left-Cerebral-White-Matter'], ["W'7", -23.05193780722757, -30.166832829691973, 21.310502628830438, 151, 106, 97, 2, 'Left-Cerebral-White-Matter'], ["W'8", -26.546587056646384, -30.341225037350593, 21.22675311274521, 154, 106, 97, 3, 'Left-Cerebral-Cortex'], ["W'9", -30.04123630606519, -30.515617245009217, 21.143003596659987, 158, 106, 97, 3, 'Left-Cerebral-Cortex'], ["W'10", -33.535885555484, -30.69000945266784, 21.05925408057476, 161, 106, 97, 3, 'Left-Cerebral-Cortex'], ["W'11", -37.03053480490281, -30.864401660326465, 20.975504564489533, 165, 107, 97, 3, 'Left-Cerebral-Cortex'], ["W'12", -40.52518405432161, -31.038793867985085, 20.891755048404306, 168, 107, 96, 3, 'Left-Cerebral-Cortex'], ["W'13", -44.01983330374042, -31.21318607564371, 20.80800553231908, 172, 107, 96, 3, 'Left-Cerebral-Cortex'], ["W'14", -47.51448255315923, -31.387578283302332, 20.724256016233852, 175, 107, 96, 2, 'Left-Cerebral-White-Matter'], ["W'15", -51.00913180257805, -31.561970490960956, 20.64050650014863, 179, 107, 96, 3, 'Left-Cerebral-Cortex']]
subject_nodes = pd.DataFrame(subject_nodes)
new_header = subject_nodes.iloc[0] #grab the first row for the header
subject_nodes = subject_nodes[1:] #take the data less the header row
subject_nodes.columns = new_header #set the header row as the df header

def is_it_the_area(x, region):
    if region == 'cortex':
        area = ['Left-Cerebral-Cortex','Right-Cerebral-Cortex','Unknown']
    elif region == 'hippo':
        area = ['Left-Hippocampus','Right-Hippocampus']
    elif region == 'amy':
        area = ['Left-Amygdala','Right-Amygdala']
    if x in  area:
        return True
    else:
        return False

region = input('Area of interest [cortex/hippo/amy]: ')
subject_nodes['cortex'] = subject_nodes['Anatomic Region'].apply(lambda x : is_it_the_area(x,region))
cortical_channels = subject_nodes[subject_nodes['cortex']==True].Node.to_list()
single_cortical_electrodes = np.unique([ ''.join((x for x in name if not x.isdigit())) for name in cortical_channels]).tolist()
all_cortical_electrodes = []
for electrode in all_electrodes:
    in_contacts = []
    for contact in electrode: 
        first_contact, second_contact = contact.split('-') 
        if first_contact in cortical_channels or second_contact in cortical_channels:
            in_contacts.append(first_contact+'-'+second_contact)
    all_cortical_electrodes.append(in_contacts)
all_cortical_electrodes = [x for x in all_cortical_electrodes if x]

if len(all_cortical_electrodes)==0:
    print('No electrodes in the '+region)

#%%

method_params = dict(diagonal_fixed=dict(seeg=0.01))

conditions = ['LaN','LaS']


for channel_set in all_cortical_electrodes:
    for condition in conditions:
        condition_epochs = epochs[condition].pick_channels(channel_set)
        noise_covs = mne.cov.compute_covariance(condition_epochs, tmin=None, tmax=0, method='auto',return_estimators=False, verbose=False, n_jobs=1, projs=None, rank=None,method_params=method_params)
        avg_epochs_evoked = condition_epochs.apply_baseline((4,5)).average(method='mean')
        title = np.unique([ ''.join((x for x in name if not x.isdigit())) for name in channel_set]).tolist()[0]+' '+condition
        avg_epochs_evoked.crop(5,11).plot(titles=title,gfp='only')
        avg_epochs_evoked.crop(5,11).plot(titles=title,noise_cov=noise_covs)
        avg_epochs_evoked.crop(5,11).plot_white(noise_cov=noise_covs)


#%%

for channels_of_interest in all_cortical_electrodes:

    print('\n')
    print(channels_of_interest)
    
    print('\n TFR')
    epochs = mne.Epochs(reconst_raw, events, event_id, baseline=None, detrend=1, tmax=11, picks=channels_of_interest, preload=True)
    freqs = np.logspace(*np.log10([1, 140]), num=30)
    n_cycles = freqs/2.  # different number of cycle per frequency
    power = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True,return_itc=False,decim=1, average=False, verbose=False)
    baseline = (4,5)
    power = power.copy().apply_baseline(baseline, mode="logratio").crop(4.9,11)
    
    
    for event in event_id:
        # select desired epochs for visualization
        power_ev = power[event]
        n_ch = len(channels_of_interest)
        if n_ch ==1 :
            power_ev.average().plot(yscale='log',cmap="jet",colorbar=True, show=False, verbose=False)
            plt.title(epochs.ch_names, fontsize=10)
            plt.axvline(5, linewidth=3, color="black", linestyle=":")  # event
            plt.axvline(8, linewidth=3, color="black", linestyle=":")  # event
            plt.suptitle(f"TFR ({event})")
            plt.show()
        else:
            fig, axes = plt.subplots(1, n_ch, figsize=(n_ch*10, 4),gridspec_kw={"width_ratios": [10]*n_ch})
            for ch, ax in enumerate(axes[:]):  # for each channel
                power_ev.average().plot([ch], yscale ='log',cmap="jet", axes=ax,colorbar=True, show=False, verbose=False)
                ax.set_title(epochs.ch_names[ch], fontsize=10)
                ax.axvline(5, linewidth=3, color="black", linestyle=":")  # event
                ax.axvline(8, linewidth=3, color="black", linestyle=":")  # event
                if ch != 0:
                    ax.set_ylabel("")
                    ax.set_yticklabels("")
            fig.suptitle(f"TFR ({event})")
            plt.show()

    n_ch = len(channels_of_interest)
    if n_ch == 1: 
        initial_contact , end_contact = channels_of_interest[0].split('-')
        initial_num = int(re.search(r'\d+', initial_contact).group())
        end_num = int(re.search(r'\d+', end_contact).group())
        letter = ''.join([i for i in initial_contact if not i.isdigit()])
        
        if (letter+str(initial_num+1)+'-'+letter+str(end_num+1)) not in all_channels.ch_names:
            channels_of_interest = [letter+str(initial_num-1)+'-'+letter+str(end_num-1)] + channels_of_interest
        else:
            channels_of_interest = channels_of_interest + [letter+str(initial_num+1)+'-'+letter+str(end_num+1)]
    n_ch = len(channels_of_interest)

    print('\nPower as a percentage')
    # Compute power again but as a percent
    epochs = mne.Epochs(reconst_raw, events, event_id, baseline=None, detrend=1, tmax=10, picks=channels_of_interest, preload=True)
    baseline = (4,5)
    power = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True,return_itc=False,decim=3, average=False)
    power = power.copy().apply_baseline(baseline, mode="percent").crop(4,9)

    print('\nERDS')
    # Clusters
    freqs = np.arange(1, 140)  # frequencies from 1-140Hz
    vmin, vmax = -1,1  # set min and max ERDS values in plot
    
    cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  # min, center & max ERDS
    kwargs = dict(n_permutations=100, step_down_p=0.05, seed=1, buffer_size=None, out_type='mask')  # for cluster test
    
    for event in event_id:
        print('for the '+event+' event')
        # select desired epochs for visualization
        power_ev = power[event]
        n_ch = len(channels_of_interest)
        fig, axes = plt.subplots(1, n_ch, figsize=(n_ch*4, 4),gridspec_kw={"width_ratios": [10]*n_ch})
        for ch, ax in enumerate(axes[:]):  # for each channel
            # positive clusters
            _, c1, p1, _ = pcluster_test(power_ev.data[:, ch], tail=1, **kwargs, verbose=False)
            # negative clusters
            _, c2, p2, _ = pcluster_test(power_ev.data[:, ch], tail=-1, **kwargs, verbose=False)
    
            # note that we keep clusters with p <= 0.05 from the combined clusters
            # of two independent tests; in this example, we do not correct for
            # these two comparisons
            c = np.stack(c1 + c2, axis=2)  # combined clusters
            p = np.concatenate((p1, p2))  # combined p-values
            mask = c[..., p <= 0.05].any(axis=-1)
    
            # plot TFR (ERDS map with masking)
            power_ev.average().plot([ch], cmap="RdBu", cnorm=cnorm, axes=ax,colorbar=True, show=False, mask=mask, mask_style="mask", verbose=False)
    
            ax.set_title(epochs.ch_names[ch], fontsize=10)
            ax.axvline(5, linewidth=1, color="black", linestyle=":")  # event
            ax.axvline(7, linewidth=1, color="black", linestyle=":")  # event
            if ch != 0:
                ax.set_ylabel("")
                ax.set_yticklabels("")
        fig.suptitle(f"ERDS ({event})")
        plt.show()


    print('\nConverting results into dataframe')
    df = power.to_data_frame(time_format=None, long_format=True)
    
    # Map to frequency bands:
    freq_bounds = {'_': 0,
                   'delta': 3,
                   'theta': 7,
                   'alpha': 13,
                   'beta': 35,
                   'gamma': 140}
    df['band'] = pd.cut(df['freq'], list(freq_bounds.values()),labels=list(freq_bounds)[1:])
    
    # Filter to retain only relevant frequency bands:
    freq_bands_of_interest = ['delta', 'theta','alpha','beta','gamma']
    df = df[df.band.isin(freq_bands_of_interest)]
    df['band'] = df['band'].cat.remove_unused_categories()
    
    df['channel'] = df['channel'].cat.reorder_categories(channels_of_interest, ordered=True)
    
    g = sns.FacetGrid(df, row='band', col='channel', margin_titles=True)
    g.map(sns.lineplot, 'time', 'value', 'condition', n_boot=10)
    axline_kw = dict(color='black', linestyle='dashed', linewidth=0.5, alpha=0.5)
    g.map(plt.axhline, y=0, **axline_kw)
    g.map(plt.axvline, x=5, **axline_kw)
    g.map(plt.axvline, x=7, **axline_kw)
    g.set_axis_labels("Time (s)", "ERDS (%)")
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.add_legend()
    g.fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.08)

