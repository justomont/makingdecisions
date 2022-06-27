"""
Created on Tue Mar  8 14:17:58 2022

@author: justo & emma
"""
import os
import h5py
from math import isnan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as opt
from scipy.ndimage import gaussian_filter as gauss
import statsmodels.formula.api as smf
import statsmodels.api as sm
import scipy.stats as sp
from textwrap import wrap
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from statsmodels.formula.api import ols
import seaborn as sns
sns.set_theme()

### FUNCTIONS ###


def closest(lst, K):
    '''
    closest will find the closest number to the one you give as an input, in another vector

    Parameters
    ----------
    lst : vector
        vector where you want to find the number.
    K : float
        number which you want to find.

    Returns
    -------
    float
        closest number to K in 1st.

    ''' 
    lst = np.asarray(lst)
    idx = (np.abs(lst - K)).argmin()
    return lst[idx]

def f(x, a, b, c, d):
    '''
    f is the logarithmic function used to fit the psychometric curve
    '''
    return a / (1. + np.exp(-c * (x - d))) + b

def select_eyedata(trial,trial_n,etData,tmin,tmax,gaze_threshold):
    '''
    select eyetracking data for a specific trial

    Parameters
    ----------
    trial : vector
        data for a specific trial.
    trial_n : int
        index of the trial.
    etData : DataFrame
        eyetracking data for all trials.
    tmin : string
        where you want to start the trial.
    tmax : string
        where you want to end the trial.
    gaze_threshold : float
        threshold to select looking at left or right.

    Returns
    -------
    DataFrame
        eyetracking data for a specific trial.

    '''
    #PROBLEMA AQUÍ
    #select eyedata for the trial
    global gazedata
    
    if isinstance(tmin, str) and isinstance(tmax, str):
        gazedata = etData.loc[(etData['time']>=trial[tmin]) & (etData['time']<=trial[tmax]), ['time','left_gaze_x', 'left_gaze_y','right_gaze_x', 'right_gaze_y','left_pupil_measure1']]
    elif isinstance(tmin, float) and isinstance(tmax, float):
        gazedata = etData.loc[(etData['time']>=tmin) & (etData['time']<=tmax), ['time','left_gaze_x', 'left_gaze_y','right_gaze_x', 'right_gaze_y']]
    else:
        print('Tmin and Tmax are not in the same format, they have to be either str,str or float,float.')
    
    # label right and left trials
    gazedata['position'] = float('nan')
    right_trials = (gazedata['left_gaze_x']>gaze_threshold) | (gazedata['right_gaze_x']>gaze_threshold)
    left_trials = (gazedata['left_gaze_x']<-gaze_threshold) | (gazedata['right_gaze_x']<-gaze_threshold)
    gazedata.loc[right_trials,'position'] = True
    gazedata.loc[left_trials,'position'] = False
    
    # remove rows with no data
    # gazedata[([isnan(i) for i in list(gazedata.left_gaze_x)]) and ([isnan(i) for i in list(gazedata.right_gaze_x)])] = float('nan')
    return(gazedata)    

def fraction_of_looking_time(gazedata, gaze_threshold, task):
    
    if 'gambling' in task:
        right_count = np.sum((gazedata['left_gaze_x']>=gaze_threshold)| (gazedata['right_gaze_x']>=gaze_threshold))
        left_count = np.sum((gazedata['left_gaze_x']<=-gaze_threshold)| (gazedata['right_gaze_x']<=-gaze_threshold))
    
    else:
        right_count = np.sum( (((gazedata['left_gaze_x']>=gaze_threshold) & (gazedata['left_gaze_x']<=1-gaze_threshold)) & ((gazedata['left_gaze_y']>=-gaze_threshold) & (gazedata['left_gaze_y']<=gaze_threshold))) | (((gazedata['right_gaze_x']>=gaze_threshold) & (gazedata['right_gaze_x']<=1-gaze_threshold)) & ((gazedata['right_gaze_y']>=-gaze_threshold) & (gazedata['right_gaze_y']<=gaze_threshold)))  )
        left_count = np.sum(  (((gazedata['left_gaze_x']<=-gaze_threshold) & (gazedata['left_gaze_x']>=-1+gaze_threshold)) & ((gazedata['left_gaze_y']>=-gaze_threshold) & (gazedata['left_gaze_y']<=gaze_threshold))) |  (((gazedata['right_gaze_x']<=-gaze_threshold) & (gazedata['right_gaze_x']>=-1+gaze_threshold)) & ((gazedata['right_gaze_y']>=-gaze_threshold) & (gazedata['right_gaze_y']<=gaze_threshold))) )
    
    if right_count==0 and left_count==0:
        R_fraction = float('nan')
        L_fraction = float('nan')
    else:
        R_fraction = right_count/(right_count+left_count)
        L_fraction = left_count/(right_count+left_count)
    
    return(R_fraction,L_fraction)

def remove_missed_trials(trials,etData):
   
    # missed_list = []
    # for index, trial in trials.iterrows():   
    #     gazedata = select_eyedata(trial,index,etData,'et_looking_Start','et_trial_End',gaze_threshold=0)
    #     R_fraction, L_fraction, missed = fraction_of_looking_time(gazedata, gaze_threshold=0)
    #     missed_list.append(missed)
    # trials['missed'] = missed_list
    # trials = trials.drop(trials[trials['missed']>0.25].index).reset_index(drop=True)
   
    return(trials)

def predict_choice_lt(gazedata,sf):
    '''
    predicts choice of the subject taking into account the looking time
    
    Parameters
    ----------
    gazedata : DataFrame
        Eyetracking data for one trial. 
    sf : float
        sampling frequency (120 in the Tobii eyetracker).

    Returns
    -------
    right_rate: float
        proportion of time looking at the right option.
    right_pred: Bool
        True if the proportion of time looking at right is > than 0.5.
    right_lt: float
        seconds looking at right.
    left_lt: float
        seconds looking at left.

    '''
    try:
        right_rate = gazedata['position'].value_counts(normalize=True)[True]
    except:
        right_rate = 0
    right_pred = right_rate>=0.5
    right_lt = sum(gazedata['position']==True)/sf
    left_lt = sum(gazedata['position']==False)/sf
    return(right_rate, right_lt, left_lt, right_pred)

def plot_corr(trials_all,x_n,y_n,task,last_phase = False):
    fig  = plt.figure() 
    corr, pvalue = spearmanr(trials_all[x_n],trials_all[y_n])    
    fig.text(0.25,0.75,"p =" + str(pvalue)+'\n corr = '+str(np.round(corr,3))) 
    plt.title(x_n+' vs '+y_n+' \n'+task+' task',fontsize=25)
    
    
    # plt.plot(trials_all[x_n],trials_all[y_n],'.')
    
    
    if last_phase == True and x_n == 'chosen_lt' :
        Y= trials_all[y_n] #the responses of the experiment
        bins = np.linspace(min(trials_all[x_n]),max(trials_all[x_n]),4)
        X1= [bins[i] for i in np.digitize(trials_all[x_n], bins)-1]

        nx = np.unique(X1)
        mn=[np.mean(Y[X1==x]) for x in nx] 
        sem=[ sp.sem(Y[X1==x]) for x in nx]
        plt.errorbar(nx*10,mn,yerr=sem, fmt="ko",capsize = 6);
        plt.plot(nx*10,mn,color='silver')
        plt.xlabel(x_n,fontsize = 20)
        plt.ylabel(y_n,fontsize = 20)
        plt.xticks(fontsize = 15)
        plt.yticks(fontsize = 15)
        
        trials_all[x_n] = [int(i*10) for i in trials_all[x_n]]
        trials_all[y_n] = [int(i) for i in trials_all[y_n]]
        # Fit and summarize OLS model
        
        mod = ols(formula = y_n+' ~ '+x_n , data=trials_all)
        res = mod.fit()
        
        
        myx = np.linspace(min(trials_all[x_n]),max(trials_all[x_n]),100)
        line_fit=res.predict(pd.DataFrame({x_n: myx}))
        plt.plot(myx,line_fit,'-',color='grey', linewidth=5,label = 'ols')
    else:
        Y= trials_all[y_n] #the responses of the experiment
        X1=[int(x) for x in trials_all[x_n]]
        # X1 = np.round(trials_all[x_n],1)
        nx = np.unique(X1)
        mn=[np.mean(Y[X1==x]) for x in nx] 
        sem=[ sp.sem(Y[X1==x]) for x in nx]
        plt.errorbar(nx,mn,yerr=sem, fmt="ko",capsize = 6);
        plt.plot(nx,mn,color='silver')
        plt.xlabel(x_n,fontsize = 20)
        plt.ylabel(y_n,fontsize = 20)
        plt.xticks(fontsize = 15)
        plt.yticks(fontsize = 15)
        
        trials_all[x_n] = trials_all[x_n].astype(int)
        trials_all[y_n] = trials_all[y_n].astype(int)
        # Fit and summarize OLS model
        
        mod = ols(formula = y_n+' ~ '+x_n , data=trials_all)
        res = mod.fit()
        
        
        myx = np.linspace(min(trials_all[x_n]),max(trials_all[x_n]),100)
        line_fit=res.predict(pd.DataFrame({x_n: myx}))
        plt.plot(myx,line_fit,'-',color='grey', linewidth=5,label = 'ols')
    return(corr,pvalue)

def get_food_data(fileDir,subj):
    '''
    get the data for the food task
    etData = pd.DataFrame(np.array(h5py.File(fileDir+[file for file in files_subject if file.endswith(r'.hdf5')][0])['data_collection']['events']['eyetracker']['BinocularEyeSampleEvent']))

    Parameters
    ----------
    fileDir : string
        directory where the data is stored.
    subj : int
        number of the subject.

    Returns
    -------
    trials: DataFrame
        experimental data of the subject.
    etData: DataFrame
        eyetracking data of the subject.
        
    '''
    global trials
    global etData
    files_subject = [_ for _ in os.listdir(fileDir) if (_.endswith(r".hdf5") or _.endswith(r".xlsx")) and '0'+str(subj)+'_' in _]
    experiment_data = pd.ExcelFile(fileDir+[file for file in files_subject if file.endswith(r'.xlsx')][0])
    ratings_data_all = pd.read_excel(experiment_data, 'ratings')
    trials_data_all = pd.read_excel(experiment_data, 'trials')
    ratings_data = ratings_data_all.iloc[:-8 , :]
    psyData = trials_data_all.iloc[:-8,:].reset_index()
    
    sel_vars = ['image_r',
                'image_l',
                'LaN_condition',
                'cross.started_mean',
                'food_1.started_mean',
                'food_1_LaN.started_mean',
                'sound_2.started_mean',
                'mouse.leftButton_mean',
                'mouse.rightButton_mean',
                'confidence_assessment.started_mean',
                'confidence_assessment.response_mean']
    
    trials = psyData.loc[:, sel_vars].reset_index(drop=True)
    new_names = {'image_l': 'stimL',
                  'image_r':'stimR',
                  'trials.thisIndex': 'real_trial_index',
                  'food_1.started_mean': 'looking_phase_start',
                  'food_1_LaN.started_mean': 'LaN_start',
                  'sound_2.started_mean': 'decision_phase_start',
                  'confidence_assessment.response_mean': 'confidence',     
                  'LaN_condition': 'LaN',
                  'index': 'real_trial_index'}
    trials = trials.rename(columns=new_names)
    
    
    # # New variables 
    trials['decision_time'] = trials['confidence_assessment.started_mean'] - trials['decision_phase_start']
    trials['response_bool'] = (trials['mouse.leftButton_mean']+trials['mouse.rightButton_mean'])==1
    trials['response'] = trials['mouse.rightButton_mean'] == 1 # Right is True
    
    # # Remove invalid trials (no choice)
    trials = trials.drop(trials[trials['response_bool']==False].index).reset_index(drop=True)
    for index, row in trials.iterrows():
        if 'kitkat' in row.stimL or 'kitkat' in row.stimR:
            if 'chips' in row.stimL or 'chips' in row.stimR:
                ind = index
    trials = trials.drop(ind)
    
    trials['stimL_value'] = [ratings_data[ratings_data.im_ratings == row.stimR].iloc[0]['rating_food.response_mean'] for index,row in trials.iterrows()]
    trials['stimR_value'] = [ratings_data[ratings_data.im_ratings == row.stimL].iloc[0]['rating_food.response_mean'] for index,row in trials.iterrows()]
    
    etData = pd.DataFrame(np.array(h5py.File(fileDir+[file for file in files_subject if file.endswith(r'.hdf5')][0])['data_collection']['events']['eyetracker']['BinocularEyeSampleEvent']))
    trials['et_looking_Start'] = [closest(list(etData['time']),t) for t in trials['looking_phase_start']]
    trials['et_LaN_Start'] = [closest(list(etData['time']),t) for t in trials['LaN_start']]
    trials['et_decision_Start'] = [closest(list(etData['time']),t) for t in trials['decision_phase_start']]
    trials['et_trial_End'] = [closest(list(etData['time']),t) for t in trials['confidence_assessment.started_mean']]
    trials['etDelay'] = [closest(list(etData['time']),t) - t for t in trials['decision_phase_start']]
    trials['value_bool'] = (trials['stimR_value']!='None') & (trials['stimL_value']!='None')
    trials = trials.drop(trials[trials['value_bool']==False].index).reset_index(drop=True)
    trials['stimR_value'] = [float(n) for n in list(trials['stimR_value'])]
    trials['stimL_value'] = [float(n) for n in list(trials['stimL_value'])]
    trials['evidence_right'] = (trials['stimR_value'] - trials['stimL_value'])
    
    # Remove invalid trials (Missed fraction of looking time over 0.25)
    # trials = remove_missed_trials(trials,etData)
    trials = remove_missed_trials(trials,etData)

    return(trials, etData)

def get_context_data(fileDir,subj):
    '''
    get the data for the context task

    Parameters
    ----------
    fileDir : string
        directory where the data is stored.
    subj : int
        number of the subject.

    Returns
    -------
    trials: DataFrame
        experimental data of the subject.
    etData: DataFrame
        eyetracking data of the subject.
        
    '''
    global trials
    global etData 
    files_subject = [_ for _ in os.listdir(fileDir) if (_.endswith(r".hdf5") or _.endswith(r".csv")) and "P00"+str(subj) in _]
    psyData = pd.read_csv(fileDir+[file for file in files_subject if file.endswith(r'.csv')][0])
    bool_trials = [isinstance(i, str) for i in list(psyData.written_context)] # select just the rows in the output file of the trials
    sel_vars = ['written_context',
                'imageA',
                'imageB',
                'LaN',
                'trials.thisIndex',
                'stimA.started',
                'stimA_3.started',
                'sound_1.started',
                'resp_choice.leftButton',
                'resp_choice.rightButton',
                'question_confidence.started',
                'slider_confidence.response',
                'slider_valueA.response',
                'slider_valueB.response']
    
    trials = psyData.loc[bool_trials, sel_vars].reset_index(drop=True)
    
    # Change names to make it more readable
    new_names = {'imageA': 'stimL',
                 'imageB':'stimR',
                 'trials.thisIndex': 'real_trial_index',
                 'stimA.started': 'looking_phase_start',
                 'stimA_3.started': 'LaN_start',
                 'slider_confidence.response': 'confidence',
                 'slider_valueA.response': 'stimL_value',
                 'slider_valueB.response': 'stimR_value',
                 'sound_1.started': 'decision_start'}
    trials = trials.rename(columns=new_names)
    
    # New variables 
    trials['decision_time'] = trials['question_confidence.started'] - trials['decision_start']
    trials['response_bool'] = (trials['resp_choice.leftButton']+trials['resp_choice.rightButton'])==1
    trials['response'] = trials['resp_choice.rightButton'] == 1 # Right is True
    
    # Remove invalid trials (no choice)
    trials = trials.drop(trials[trials['response_bool']==False].index).reset_index(drop=True)
    
    # Eyetracking data
    etData = pd.DataFrame(np.array(h5py.File(fileDir+[file for file in files_subject if file.endswith(r'.hdf5')][0])['data_collection']['events']['eyetracker']['BinocularEyeSampleEvent']))

    trials['et_looking_Start'] = [closest(list(etData['time']),t) for t in trials['looking_phase_start']]
    trials['et_LaN_Start'] = [closest(list(etData['time']),t) for t in trials['LaN_start']]
    trials['et_decision_Start'] = [closest(list(etData['time']),t) for t in trials['decision_start']]
    trials['et_trial_End'] = [closest(list(etData['time']),t) for t in trials['question_confidence.started']]
    trials['etDelay'] = [closest(list(etData['time']),t) - t for t in trials['decision_start']]
    trials['value_bool'] = (trials['stimR_value']!='None') & (trials['stimL_value']!='None')
    # trials = trials.drop(trials[trials['value_bool']==False].index).reset_index(drop=True)
    trials['stimR_value'] = [float(n) for n in list(trials['stimR_value'])]
    trials['stimL_value'] = [float(n) for n in list(trials['stimL_value'])]
    trials['evidence_right'] = (trials['stimR_value'] - trials['stimL_value'])
    
    # Remove invalid trials (Missed fraction of looking time over 0.25)
    # trials = remove_missed_trials(trials,etData)

    return(trials, etData)
 
def get_gambling_data(fileDir,subj):
    '''
    get the data for the gambling task

    Parameters
    ----------
    fileDir : string
        directory where the data is stored.
    subj : int
        number of the subject.

    Returns
    -------
    trials: DataFrame
        experimental data of the subject.
    etData: DataFrame
        eyetracking data of the subject.
        
    '''
    global trials
    global etData
    files_subject = [_ for _ in os.listdir(fileDir) if (_.endswith(r".hdf5") or _.endswith(r".xlsx")) and '0'+str(subj)+'_' in _]
    # experiment_data = pd.read_csv(fileDir+[file for file in files_subject if file.endswith(r'.csv')][0])
    # global psyData
    # psyData = experiment_data.iloc[19:,:].reset_index()
    # psyData = psyData.iloc[:-1,:].reset_index()
    
    
    experiment_data = pd.ExcelFile(fileDir+[file for file in files_subject if file.endswith(r'.xlsx')][0])
    trials_data_all = pd.read_excel(experiment_data, 'trials')
    psyData = trials_data_all.iloc[:-8,:].reset_index()

    
    sel_vars = ['heightl',
                'heightr',
                'colorl',
                'colorr',
                'LaN_condition',
                'fix.started_mean',
                'cl_LAN.started_mean',
                'rl.started_mean',
                'sound_2.started_mean',
                'mouse_7.leftButton_mean',
                'mouse_7.rightButton_mean',
                'slider.started_mean',
                'slider.response_mean',
                'index']
    
    trials = psyData.loc[:, sel_vars].reset_index(drop=True)
      
    # Rename variables
    new_names = {'heightl': 'stimL',
                  'heightr':'stimR',
                  'index': 'real_trial_index',
                  'fix.started_mean': 'looking_phase_start',
                  'cl_LAN.started_mean': 'LaN_phase_start',
                  'slider.response_mean': 'confidence',     
                  'rl.started_mean': 'looking_2imgs_start',
                  'sound_2.started_mean': 'decision_phase_start',
                  'mouse_7.leftButton_mean':'mouse.leftButton',
                  'mouse_7.rightButton_mean':'mouse.rightButton',
                  'LaN_condition': 'LaN'}
    trials = trials.rename(columns=new_names)
    # New variables 
    trials.loc[trials['colorl']=='lightgrey','colorl'] = 5 
    trials.loc[trials['colorl']=='blue','colorl'] = 7 
    trials.loc[trials['colorl']=='green','colorl'] = 10
    trials.loc[trials['colorr']=='lightgrey','colorr'] = 5 
    trials.loc[trials['colorr']=='blue','colorr'] = 7 
    trials.loc[trials['colorr']=='green','colorr'] = 10
    trials['decision_time'] = trials['slider.started_mean'] - trials['decision_phase_start']
    trials['response_bool'] = (trials['mouse.leftButton']+trials['mouse.rightButton'])==1
    trials['response'] = trials['mouse.rightButton'] == 1 # Right is True
    # ACABAR
    trials['stimL_value'] = trials.colorl*trials.stimL.astype(float)*2
    trials['stimR_value'] = trials.colorr*trials.stimR.astype(float)*2
    global trials2
    trials2 = trials
    # trials['stimL_value'] = [trials.colorl*trials.stimL*2 for index,row in trials.iterrows()]
    # trials['stimR_value'] = [trials.colorr*trials.stimR*2 for index,row in trials.iterrows()]
    
    # # Remove invalid trials (no choice)
    trials = trials.drop(trials[trials['response_bool']==False].index).reset_index(drop=True)
  
    
    etData = pd.DataFrame(np.array(h5py.File(fileDir+[file for file in files_subject if file.endswith(r'.hdf5')][0])['data_collection']['events']['eyetracker']['BinocularEyeSampleEvent']))
    trials['et_looking_Start'] = [closest(list(etData['time']),t) for t in trials['looking_phase_start']]
    trials['et_LaN_Start'] = [closest(list(etData['time']),t) for t in trials['LaN_phase_start']]
    trials['et_decision_Start'] = [closest(list(etData['time']),t) for t in trials['decision_phase_start']]
    trials['et_trial_End'] = [closest(list(etData['time']),t) for t in trials['slider.started_mean']]
    trials['etDelay'] = [closest(list(etData['time']),t) - t for t in trials['decision_phase_start']]
    trials['value_bool'] = (trials['stimR_value']!='None') & (trials['stimL_value']!='None')
    trials = trials.drop(trials[trials['value_bool']==False].index).reset_index(drop=True)
    # trials['stimR_value'] = [float(n) for n in list(trials['stimR_value'])]
    # trials['stimL_value'] = [float(n) for n in list(trials['stimL_value'])]
    trials['evidence_right'] = (trials['stimR_value'] - trials['stimL_value'])

    # Remove invalid trials (Missed fraction of looking time over 0.25)
    trials = remove_missed_trials(trials,etData)

    return(trials, etData)

def get_gambling_data_H(fileDir,subj):
    '''
    get the data for the gambling task

    Parameters
    ----------
    fileDir : string
        directory where the data is stored.
    subj : int
        number of the subject.

    Returns
    -------
    trials: DataFrame
        experimental data of the subject.
    etData: DataFrame
        eyetracking data of the subject.
        
    '''
    global trials
    global etData
    if subj>9:
        files_subject = [_ for _ in os.listdir(fileDir) if (_.endswith(r".hdf5") or _.endswith(r".csv")) and "P0"+str(subj) in _]
    else:
        files_subject = [_ for _ in os.listdir(fileDir) if (_.endswith(r".hdf5") or _.endswith(r".csv")) and "P00"+str(subj) in _]
    experiment_data = pd.read_csv(fileDir+[file for file in files_subject if file.endswith(r'.csv')][0])

    psyData = experiment_data.iloc[:-7,:].reset_index()
    
    psyData.columns = psyData.columns.map(lambda x: x.removesuffix("_mean"))
    
    sel_vars = ['heightl',
                'heightr',
                'colorl',
                'colorr',
                'fix.started',
                'rl1.started',
                'image_2.started',
                'rr1.started',
                'image_3.started',
                'rl.started',
                'decr_2.started',
                'sound_2.started',
                'mouse_7.leftButton',
                'mouse_7.rightButton',
                'slider.started',
                'slider.response',
                'index']
    
    trials = psyData.loc[:, sel_vars].reset_index(drop=True)
      
    # Rename variables
    new_names = {'heightl': 'stimL',
                  'heightr':'stimR',
                  'index': 'real_trial_index',
                  'fix.started': 'looking_phase_start',
                  'rl1.started': 'left_offer_start',
                  'image_2.started': 'left_offer_end',
                  'rr1.started': 'right_offer_start',
                  'image_3.started': 'right_offer_end',
                  'slider.response': 'confidence',     
                  'rl.started': 'looking_2imgs_start',
                  'decr_2.started': 'pre_decision_phase_start',
                  'sound_2.started': 'decision_phase_start',
                  'mouse_7.leftButton': 'mouse.leftButton',
                  'mouse_7.rightButton': 'mouse.rightButton'}
    trials = trials.rename(columns=new_names)
    # New variables 
    trials.loc[trials['colorl']=='lightgrey','colorl'] = 5 
    trials.loc[trials['colorl']=='blue','colorl'] = 7 
    trials.loc[trials['colorl']=='green','colorl'] = 10
    trials.loc[trials['colorr']=='lightgrey','colorr'] = 5 
    trials.loc[trials['colorr']=='blue','colorr'] = 7 
    trials.loc[trials['colorr']=='green','colorr'] = 10
    trials['decision_time'] = trials['slider.started'] - trials['decision_phase_start']
    trials['response_bool'] = (trials['mouse.leftButton']+trials['mouse.rightButton'])==1
    trials['response'] = trials['mouse.rightButton'] == 1 # Right is True
    trials['stimL_value'] = trials.colorl*trials.stimL.astype(float)*2
    trials['stimR_value'] = trials.colorr*trials.stimR.astype(float)*2
    
    # Remove invalid trials (no choice)
    trials = trials.drop(trials[trials['response_bool']==False].index).reset_index(drop=True)
    
    etData = pd.DataFrame(np.array(h5py.File(fileDir+[file for file in files_subject if file.endswith(r'.hdf5')][0])['data_collection']['events']['eyetracker']['BinocularEyeSampleEvent']))
    trials['et_looking_Start'] = [closest(list(etData['time']),t) for t in trials['looking_phase_start']]
    trials['et_left_offer_Start'] = [closest(list(etData['time']),t) for t in trials['left_offer_start']]
    trials['et_left_offer_End'] = [closest(list(etData['time']),t) for t in trials['left_offer_end']]
    trials['et_right_offer_Start'] = [closest(list(etData['time']),t) for t in trials['right_offer_start']]
    trials['et_right_offer_End'] = [closest(list(etData['time']),t) for t in trials['right_offer_end']]
    trials['et_both_offers_Start'] = [closest(list(etData['time']),t) for t in trials['looking_2imgs_start']]
    trials['et_LaN_Start'] = [closest(list(etData['time']),t) for t in trials['pre_decision_phase_start']]
    trials['et_decision_Start'] = [closest(list(etData['time']),t) for t in trials['decision_phase_start']]
    trials['et_trial_End'] = [closest(list(etData['time']),t) for t in trials['slider.started']]
    trials['etDelay'] = [closest(list(etData['time']),t) - t for t in trials['decision_phase_start']]
    trials['value_bool'] = (trials['stimR_value']!='None') & (trials['stimL_value']!='None')
    trials = trials.drop(trials[trials['value_bool']==False].index).reset_index(drop=True)

    trials['evidence_right'] = (trials['stimR_value'] - trials['stimL_value'])
    
    # Remove invalid trials (Missed fraction of looking time over 0.25)
    # trials = remove_missed_trials(trials,etData)
    

    return(trials,etData)
       
### CLASSES ###
class subject:
    
    def __init__(self,fileDir,subj,tmin = 'et_looking_Start', tmax= 'et_trial_End', gaze_threshold = 0.4,sf=120,magF = 100, LaN=False, LaS=False,chosen_option = None):
    
        self.fileDir = fileDir
        self.subj = subj
        self.tmin = tmin
        self.tmax = tmax
        self.gaze_threshold = gaze_threshold
        self.sf = sf
        self.magF = magF
        self.LaN = LaN
        self.LaS = LaS
                
        print('Initializing Subject: '+str(self.subj)+'...')
        
        if 'food' in fileDir:
            self.task ='food'
            self.trials, self.etData = get_food_data(fileDir,subj)
        if 'gambling_HEALTHY' in fileDir:
            self.task ='gambling'
            self.trials, self.etData = get_gambling_data(fileDir,subj)
        if 'gambling_HOSPITAL' in fileDir:
            self.task ='gambling_hospital'
            self.trials, self.etData = get_gambling_data_H(fileDir,subj)
        if 'context' in fileDir:
            self.task ='context'
            self.trials, self.etData = get_context_data(fileDir,subj)

        # select the data for look at nothing trials or look at something trials
        if LaN:
            self.trials = self.trials.drop(self.trials[self.trials['LaN']==1].index).reset_index(drop=True)
        if LaS:
            self.trials = self.trials.drop(self.trials[self.trials['LaN']==0].index).reset_index(drop=True)   
    
    
        if chosen_option == 'right':
            self.trials = self.trials[self.trials.response == True]
        if chosen_option == 'left':
            self.trials = self.trials[self.trials.response == False]
            
    
    
        self.mean_decision_time = self.trials.decision_time.mean()
        self.std_mean_decision_time = self.trials.decision_time.std()
        self.max_decision_time = self.trials.decision_time.max()
        
        def accuracy(self):
            '''
            RETURNS
            -------
            accuracy : float
                accuracy of the prediction.

            '''
            gaze_threshold = self.gaze_threshold
            tmin = self.tmin
            tmax = self.tmax
            sf = self.sf
            
                  
            RightRate = []
            RightPred = []      
            
            for index, trial in self.trials.iterrows():
                
                # select eyetracking data for the trial:      
                gazedata = select_eyedata(trial,index,etData,tmin,tmax,gaze_threshold)           
                # Predict choice:
                right_rate, right_lt, left_lt, right_pred = predict_choice_lt(gazedata,sf)
                RightRate.append(right_rate)
                RightPred.append(right_pred)
                self.trials.loc[index,'right_lt'] = right_lt
                self.trials.loc[index,'left_lt'] = left_lt
                if trial.response == True:
                    self.trials.loc[index,'chosen_lt'] = right_lt
                if trial.response == False:
                    self.trials.loc[index,'chosen_lt'] = left_lt
            self.trials['RightRate'] = RightRate
            self.trials['RightPred'] = RightPred
            self.trials['Hits'] = self.trials['response'] == self.trials['RightPred']
            self.accuracy = sum(self.trials.Hits)/self.trials.Hits.shape[0]
            return accuracy
        transit = []
        for index, trial in self.trials.iterrows():
            global gazedata1

            # select eyetracking data for the trial:      
            gazedata1 = select_eyedata(trial,index,etData,tmin,tmax,0.2)
            # label right and left trials
            
            data_bool = [~np.isnan(i) for i in gazedata1.position]
            gazedata1 = gazedata1[np.array(data_bool)]
            transition_count = (np.diff(gazedata1['position'])!=0).sum()
            transit.append(transition_count)
            
        self.trials['transitions'] = transit
        self.trials = self.trials[self.trials['transitions']<13]
        self.trials = self.trials[self.trials['transitions']>=1]

        self.trials['stimL_value'] = [float(i) for i in self.trials['stimL_value']]
        self.trials['stimR_value'] = [float(i) for i in self.trials['stimR_value']]
        self.trials['total_value'] = self.trials['stimL_value'] + self.trials['stimR_value']
        self.trials['total_value'] = [float(i) for i in self.trials['total_value']]
        self.trials['evidence_right'] = [float(i) for i in self.trials['evidence_right']]
        self.trials['difficulty'] = [abs(i) for i in self.trials['evidence_right']]

        self.trials['transitions'] = [float(i) for i in self.trials['transitions']]
        bool_conf = trials['confidence'] != 'None'
        self.trials = self.trials[bool_conf]
        self.trials['confidence'] = [float(i) for i in self.trials['confidence']]
        accuracy(self)
        
        consistency = []
        for indx, trial in self.trials.iterrows():
            if int(trial['evidence_right'])>= 0 and trial['response'] ==True:
                consistency.append(1)
            elif int(trial['evidence_right'])< 0 and trial['response'] == False:
                consistency.append(1)
            else:
                consistency.append(0)
        self.trials['consistency'] = consistency
        
    def __repr__(self):
        text = 'Subject '+str(self.subj)+' in the '+str(self.task)+' task, from '+str(self.tmin)+ ' to '+str(self.tmax)+', with LaN =  '+str(self.LaN)+' LaS = '+str(self.LaS)
        return(text)
    
    def heatmap(self, section = None ,chosen_option = None, LaN=None, LaS=None, tmin=None, tmax=None, confidence_level=None, difficulty_level=None, bins=None):
        '''
        heatmap for all the trials        

        Parameters
        ----------
        chosen_option : string, optional
            'right' or 'left'. The default is None.
            
        section: string, optional
            'right', 'left', 'both' or 'decision'. The default is None.
            When specified, it only plots the heatmap for a specific section of the trial.
            For the gambling task (hospital version):
                'right': right offer presentation (alone).
                'left': left offer presentation (alone).
                'both': observation phase of both stim (before sound).
                'decision': decision phase of both stim (after sound).
            For the context task:
            
            For the food task:
        
        confidence: Boolean (True/False), optional
            When 'True' it returns different heatmaps, one per level of confidence of the trials. The default is 'False'.
            
        difficulty: Boolean (True/False), optional
            When 'True' it returns different heatmaps, one per level of difficulty of the trials. The default is 'False'.  
            
        Returns
        -------
        gaze_mat: matrix
            matrix of the heatmap and plot the heatmap.

        '''
        print('Heatmap. Section: '+str(section)+'. Chosen: '+str(chosen_option)+'. Subject:'+str(subj))
        
        magF = self.magF
        gaze_threshold = self.gaze_threshold
        LookData = []
        
        if not(tmin and tmax):
            if self.task == 'gambling_hospital':
                if section:
                    if section == 'right':
                        tmin = 'et_right_offer_Start'
                        tmax = 'et_right_offer_End'
                    if section == 'left':
                        tmin = 'et_left_offer_Start'
                        tmax = 'et_left_offer_End'
                    if section == 'both':
                        tmin = 'et_both_offers_Start'
                        tmax = 'et_LaN_Start'
                    if section == 'observation':
                        tmin = 'et_LaN_Start'
                        tmax = 'et_decision_Start'
                    if section == 'decision':
                        tmin = 'et_decision_Start'
                        tmax = 'et_trial_End'
    
            elif self.task == 'food' or self.task == 'context':
                if section:
                    if section == 'presentation':
                        tmin = 'et_looking_Start'
                        tmax = 'et_LaN_Start'
                    if section == 'observation':
                        tmin = 'et_LaN_Start'
                        tmax = 'et_decision_Start'
                    if section == 'decision':
                        tmin = 'et_decision_Start'
                        tmax = 'et_trial_End'
            else:
                tmin = self.tmin
                tmax = self.tmax
            
        if LaN or LaS:
            if LaN:
                heat_trials = trials.drop(trials[trials['LaN']==1].index).reset_index(drop=True)
            if LaS:
                heat_trials = trials.drop(trials[trials['LaN']==0].index).reset_index(drop=True)
        else:
            heat_trials = trials
            
        if confidence_level or confidence_level==0:
            heat_trials = trials.drop(trials[trials['confidence']!=confidence_level].index).reset_index(drop=True)
            print('confidence level:'+str(confidence_level))
            print(len(heat_trials))
            
        if difficulty_level or difficulty_level==0: 
            trials['closest_bin'] = [closest(bins,trial['evidence_right']) for index,trial in trials.iterrows()]
            heat_trials = trials.drop(trials[trials['closest_bin']!=difficulty_level].index).reset_index(drop=True)
            
            print('diff level:'+str(difficulty_level))
            print(len(heat_trials))
        
        global gaze_mat
        gaze_mat = np.zeros((magF*2,magF*2))
        
        for index, trial in heat_trials.iterrows():
            
            # select eyetracking data for the trial:      
            gazedata = select_eyedata(trial,index,etData,tmin,tmax,gaze_threshold)
            #create heatmap matrix for that trial
            if chosen_option:
                if chosen_option == 'right':
                    if trial.response == True:
                        gazedata['x'] = round(gazedata[['left_gaze_x', 'right_gaze_x']].mean(axis=1)*magF)+magF
                        gazedata['y'] = round(gazedata[['left_gaze_y', 'right_gaze_y']].mean(axis=1)*magF)+magF
                
                if chosen_option == 'left':
                    if trial.response == False:
                        gazedata['x'] = round(gazedata[['left_gaze_x', 'right_gaze_x']].mean(axis=1)*magF)+magF
                        gazedata['y'] = round(gazedata[['left_gaze_y', 'right_gaze_y']].mean(axis=1)*magF)+magF
                    
            else:
                gazedata['x'] = round(gazedata[['left_gaze_x', 'right_gaze_x']].mean(axis=1)*magF)+magF
                gazedata['y'] = round(gazedata[['left_gaze_y', 'right_gaze_y']].mean(axis=1)*magF)+magF
                
            try:
                for index, row in gazedata.iterrows(): 
                    if (isnan(row.x)==False and isnan(row.y)==False) :
                            try: gaze_mat[int(row.y),int(row.x)] += 1
                            except: pass       
            except: pass
            LookData.append(gazedata)

        # Graphical representation of the heatmaps
        gaze_mat = gauss(gaze_mat,5)
        # Plot the stimuli location for the gambling task:
        if self.task == 'gambling_hospital' or self.task == 'gambling':
            size_stim_base = 0.3*magF
            size_stim_height = 0.5*magF
            rectangle = plt.Rectangle((2*magF/3-size_stim_base/2,magF-size_stim_height/2), size_stim_base, size_stim_height,ec="green",fill=False)
            plt.gca().add_patch(rectangle)
            rectangle = plt.Rectangle(((4/3)*magF-size_stim_base/2,magF-size_stim_height/2), size_stim_base, size_stim_height,ec="green",fill=False)
            plt.gca().add_patch(rectangle)
        # Stimuli location for the food and the context task:
        else:
            size_stim = 0.25*magF
            rectangle = plt.Rectangle((magF/2-size_stim/2,magF-size_stim/2), size_stim, size_stim,ec="green",fill=False)
            plt.gca().add_patch(rectangle)
            rectangle = plt.Rectangle(((3/2)*magF-size_stim/2,magF-size_stim/2), size_stim, size_stim,ec="green",fill=False)
            plt.gca().add_patch(rectangle)
        plt.imshow(gaze_mat, cmap='hot', interpolation='nearest')
        # if difficulty_level or difficulty_level==0:         
        #     plt.title(str(difficulty_level))
        plt.plot(magF,magF,'w+')
        plt.title('Stage: '+section+'. Chosen:'+chosen_option+'. \nTask: '+str(self.task)+'. Subject: '+str(subj)+'.')
        plt.axis('off')
        plt.show()
        return(gaze_mat)
    
    def heatmaps(self, confidence=False, difficulty=False ):
        '''
        heatmaps for all the trials according to confidence or difficulty     

        Parameters
        ----------
        confidence: Boolean (True/False), optional
            When 'True' it returns different heatmaps, one per level of confidence of the trials. The default is 'False'.
            
        difficulty: Boolean (True/False), optional
            When 'True' it returns different heatmaps, one per level of difficulty of the trials. The default is 'False'.  
            
        Returns
        -------
        gaze_mat: matrix
            matrix of the heatmap and plot the heatmap.

        '''
        trials = self.trials
        
        if difficulty == True:    
            # bins = np.linspace(min(trials['evidence_right'])-1,max(trials['evidence_right']+1),11)  
            bins = np.linspace(min(trials['evidence_right'])-1,max(trials['evidence_right']+1),11)[1:-1] 
            for idx, level in enumerate(bins): 
               self.heatmap(section='both',chosen_option ='right',difficulty_level=level,bins=bins)
 
        if confidence == True:
            bins = list(set(trials.confidence))
            for level in bins: 
                self.heatmap(section='observation',chosen_option ='right',confidence_level=level)
    
    def psychometric_curve(self):
        '''
        plots the psychometric curve for the subject

        '''
        trials = self.trials
      
        if 'gambling' not in self.task:
            bins = np.linspace(-4,4,10) 
            labels = np.round(np.linspace(-4,4,9) ,2) 
            
            # trials['binned'] = pd.cut(trials['evidence_right'], bins=bins,labels = labels)
            valuesRL = np.unique(trials.evidence_right)
            ratiosR = []
            total_number_trials = []
            sems = []
            for valueRL in valuesRL:
                if sum(trials['evidence_right'] == valueRL) == 0:
                    ratioR = 0.0000000000001
                    total_number_trials.append(float(0.0001))
                    ratiosR.append(ratioR)
                else:
                    ratioR = np.nanmean(trials[trials['evidence_right'] == valueRL].response,axis=0)
                    sem = np.nanstd(trials[trials['evidence_right'] == valueRL].response,axis=0)/np.sqrt(len(trials[trials['evidence_right'] == valueRL].response))
                    ratioR = sum((trials['evidence_right'] == valueRL) & (trials['response']==True))/sum(trials['evidence_right'] == valueRL)
                    total_number_trials.append(sum(trials['evidence_right'] == valueRL))
                    ratiosR.append(ratioR)
                    sems.append(sem)
            
            print(sems)
            print(valuesRL)
                    
            x = valuesRL
            y = ratiosR
      
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.plot(x, y, 'ko', label='Data points')
            
            #Confidence intervals
            # z = 1.96
            # conf_int = [[y[i]-z*(np.sqrt((y[i]*(1-y[i]))/total_number_trials[i])) for i in range(len(total_number_trials))],
            #             [y[i]+z*(np.sqrt((y[i]*(1-y[i]))/total_number_trials[i])) for i in range(len(total_number_trials))]]
            # conf_int = [[y[i]-conf_int[0][i] for i in range(len(y))],[conf_int[1][i]-y[i] for i in range(len(y))]]
            # plt.errorbar(valuesRL,y,conf_int,marker='o',color = 'k',capsize = 6, linewidth=3)
            plt.plot(valuesRL,y,'k')
            plt.xlim(min(labels),max(labels))
            psycho_sem=[ sp.sem(trials.response[trials['evidence_right']==x]) for x in valuesRL]
            plt.errorbar(valuesRL,ratiosR,yerr=psycho_sem, fmt="ko",capsize = 6)
            plt.fill_between(valuesRL, [sum(value) for value in zip(ratiosR, sems)], [valueA-valueB for valueA, valueB in zip(ratiosR, sems)],alpha=0.5,label = 'SEM')
            plt.hlines(0.5,-10,10,color = 'red',alpha = 0.5)
            plt.vlines(0,0,1,color = 'red',alpha = 0.5)
        
            
            
            #Logistic fit

            trials['n_response'] = trials.response*1

            mod = smf.glm(formula= 'n_response ~ evidence_right' , data= trials , family= sm.families.Binomial() )
            res = mod.fit()

            # plt.figure(figsize=(6, 6), dpi=300)

            myx = np.linspace(min(trials['evidence_right']),max(trials['evidence_right']),100)
            line_fit=res.predict(pd.DataFrame({'evidence_right': myx}))
            plt.plot(myx,line_fit,'-',color='grey', linewidth=5,label = 'glm')
            
            yfit = res.fittedvalues
            
            plt.grid(visible=True,linewidth = 2,color = 'lightgrey')
            plt.title('Psychometric curve: '+str(self.subj),fontsize = 30)
            plt.xlabel('V(R)-V(L)',fontsize = 30)
            plt.ylabel('% right choice',fontsize = 30)
            plt.ylim(0,1)
            plt.xticks(fontsize = 25)
            plt.yticks(fontsize = 25)
            plt.legend(fontsize = 25)        
            plt.show()   
            
            

        else: 
            bins = np.linspace(-10,10,12) 
            labels = np.round(np.linspace(-10,10,11) ,2) 
            trials['binned'] = pd.cut(trials['evidence_right'], bins=bins,labels = labels)
            valuesRL = np.unique(trials.binned)
            ratiosR = []
            total_number_trials = []
            sems = []
            for valueRL in valuesRL:
                if sum(trials['binned'] == valueRL) == 0:
                    ratioR = 0.0000000000001
                    total_number_trials.append(float(0.0001))
                    ratiosR.append(ratioR)
                else:
                    ratioR = np.nanmean(trials[trials['binned'] == valueRL].response,axis=0)
                    sem = np.nanstd(trials[trials['binned'] == valueRL].response,axis=0)/np.sqrt(len(trials[trials['binned'] == valueRL].response))
                    ratioR = sum((trials['binned'] == valueRL) & (trials['response']==True))/sum(trials['binned'] == valueRL)
                    total_number_trials.append(sum(trials['binned'] == valueRL))
                    ratiosR.append(ratioR)
                    sems.append(sem)
            
            print(sems)
            print(valuesRL)
                    
            x = valuesRL
            y = ratiosR
      
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.plot(x, y, 'ko', label='Data points')
            
            #Confidence intervals
            # z = 1.96
            # conf_int = [[y[i]-z*(np.sqrt((y[i]*(1-y[i]))/total_number_trials[i])) for i in range(len(total_number_trials))],
            #             [y[i]+z*(np.sqrt((y[i]*(1-y[i]))/total_number_trials[i])) for i in range(len(total_number_trials))]]
            # conf_int = [[y[i]-conf_int[0][i] for i in range(len(y))],[conf_int[1][i]-y[i] for i in range(len(y))]]
            # plt.errorbar(valuesRL,y,conf_int,marker='o',color = 'k',capsize = 6, linewidth=3)
            plt.plot(valuesRL,y,'k')
            plt.xlim(min(labels),max(labels))
            psycho_sem=[ sp.sem(trials.response[trials['binned']==x]) for x in valuesRL]
            plt.errorbar(valuesRL,ratiosR,yerr=psycho_sem, fmt="ko",capsize = 6)
            plt.fill_between(valuesRL, [sum(value) for value in zip(ratiosR, sems)], [valueA-valueB for valueA, valueB in zip(ratiosR, sems)],alpha=0.5,label = 'SEM')
            plt.hlines(0.5,-10,10,color = 'red',alpha = 0.5)
            plt.vlines(0,0,1,color = 'red',alpha = 0.5)
        
            
            
            #Logistic fit
    
            trials['n_response'] = trials.response*1
    
            mod = smf.glm(formula= 'n_response ~ evidence_right' , data= trials , family= sm.families.Binomial() )
            res = mod.fit()
    
            # plt.figure(figsize=(6, 6), dpi=300)
    
            myx = np.linspace(min(trials['evidence_right']),max(trials['evidence_right']),100)
            line_fit=res.predict(pd.DataFrame({'evidence_right': myx}))
            plt.plot(myx,line_fit,'-',color='grey', linewidth=5,label = 'glm')
            
            yfit = res.fittedvalues
            
            plt.grid(visible=True,linewidth = 2,color = 'lightgrey')
            plt.title('Psychometric curve: '+str(self.subj),fontsize = 30)
            plt.xlabel('V(R)-V(L)',fontsize = 30)
            plt.ylabel('% right choice',fontsize = 30)
            plt.ylim(0,1)
            plt.xticks(fontsize = 25)
            plt.yticks(fontsize = 25)
            plt.legend(fontsize = 25)        
            plt.show()   
    
    def psycho_justo(self):
        
        trials = self.trials
        
        trials['n_response'] = trials.response*1

        mod = smf.glm(formula= 'n_response ~ evidence_right' , data= trials , family= sm.families.Binomial() )
        res = mod.fit()

        # plt.figure(figsize=(6, 6), dpi=300)

        myx = np.linspace(min(trials['evidence_right']),max(trials['evidence_right']),100)
        line_fit=res.predict(pd.DataFrame({'evidence_right': myx}))
        plt.plot(myx,line_fit,'-',color='grey', linewidth=5,label = 'glm')
        yfit = res.fittedvalues
        plt.grid(visible=True,linewidth = 2,color = 'lightgrey')
        plt.title('Psychometric curve: '+str(self.subj),fontsize = 30)
        plt.xlabel('V(R)-V(L)',fontsize = 30)
        plt.ylabel('% right choice',fontsize = 30)
        plt.ylim(0,1)
        plt.xticks(fontsize = 25)
        plt.yticks(fontsize = 25)
        plt.legend(fontsize = 25)        
        plt.show()  
        
    
    def fraction_time(self):
        '''
        plots the psychometric curve for the subject

        '''
        trials = self.trials
        trials.right_lt = [float(i) for i in trials.right_lt]
        bins = np.linspace(min(trials['right_lt']),max(trials['right_lt']),6) 
        labels = np.round(np.linspace(min(trials['right_lt']),max(trials['right_lt']),5) ,2) 
        trials['binned'] = pd.cut(trials['right_lt'], bins=bins,labels = labels)
        valuesRL = np.unique(trials.binned)
        ratiosR = []
        total_number_trials = []
        sems = []
        for valueRL in valuesRL:
            if sum(trials['binned'] == valueRL) == 0:
                ratioR = float(0.0000000000001)
                sem = np.nanstd(trials[trials['binned'] == valueRL].response,axis=0)/np.sqrt(len(trials[trials['binned'] == valueRL].response))
                sems.append(sem)

                total_number_trials.append(float(0.0001))
                ratiosR.append(ratioR)
            else:
                ratioR = np.nanmean(trials[trials['binned'] == valueRL].response,axis=0)
                sem = np.nanstd(trials[trials['binned'] == valueRL].response,axis=0)/np.sqrt(len(trials[trials['binned'] == valueRL].response))
                ratioR = sum((trials['binned'] == valueRL) & (trials['response']==True))/sum(trials['binned'] == valueRL)
                total_number_trials.append(sum(trials['binned'] == valueRL))
                ratiosR.append(ratioR)
                sems.append(sem)
        
        print(sems)
        print(valuesRL)
                
        x = valuesRL
        y = ratiosR
  
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.plot(x, y, 'ko', label='Data points')
        
        #Confidence intervals
        # z = 1.96
        # conf_int = [[y[i]-z*(np.sqrt((y[i]*(1-y[i]))/total_number_trials[i])) for i in range(len(total_number_trials))],
        #             [y[i]+z*(np.sqrt((y[i]*(1-y[i]))/total_number_trials[i])) for i in range(len(total_number_trials))]]
        # conf_int = [[y[i]-conf_int[0][i] for i in range(len(y))],[conf_int[1][i]-y[i] for i in range(len(y))]]
        # plt.errorbar(valuesRL,y,conf_int,marker='o',color = 'k',capsize = 6, linewidth=3)
        plt.plot(valuesRL,y,'k')
        plt.xlim(min(labels),max(labels))
        psycho_sem=[ sp.sem(trials.response[trials['binned']==x]) for x in valuesRL]
        plt.errorbar(valuesRL,ratiosR,yerr=psycho_sem, fmt="ko",capsize = 6)
        plt.fill_between(valuesRL, [sum(value) for value in zip(ratiosR, sems)], [valueA-valueB for valueA, valueB in zip(ratiosR, sems)],alpha=0.5,label = 'SEM')
        plt.hlines(0.5,-10,10,color = 'red',alpha = 0.5)
        plt.vlines(0,0,1,color = 'red',alpha = 0.5)
    
        
        
        #Logistic fit

        trials['n_response'] = trials.response*1

        mod = smf.glm(formula= 'n_response ~ right_lt' , data= trials , family= sm.families.Binomial() )
        res = mod.fit()

        # plt.figure(figsize=(6, 6), dpi=300)

        myx = np.linspace(min(trials['right_lt']),max(trials['right_lt']),100)
        line_fit=res.predict(pd.DataFrame({'right_lt': myx}))
        plt.plot(myx,line_fit,'-',color='grey', linewidth=5,label = 'glm')
        
        yfit = res.fittedvalues
        
        plt.grid(visible=True,linewidth = 2,color = 'lightgrey')
        plt.title('Fraction of looking time: '+str(self.subj),fontsize = 30)
        plt.xlabel('Fraction of looking right',fontsize = 30)
        plt.ylabel('% right choice',fontsize = 30)
        plt.ylim(0,1)
        plt.xticks(fontsize = 25)
        plt.yticks(fontsize = 25)
        plt.legend(fontsize = 25)        
        plt.show() 
    
    def plot_eyedata_trials(self):
        '''
        plots the eyetracking data for each trial

        '''
        etData = self.etData
        tmin = self.tmin
        tmax = self.tmax
        gaze_threshold = self.gaze_threshold
        sf = self.sf
        trials = self.trials
        for index, trial in trials.iterrows():
            if index == 32:
                # select eyetracking data for the trial:      
                gazedata = select_eyedata(trial,index,etData,tmin,tmax,gaze_threshold)
                   
                # Predict choice:
                right_rate, right_lt, left_lt, right_pred = predict_choice_lt(gazedata,sf)
               
                plt.plot(gazedata['left_gaze_x'])
                plt.plot(gazedata['right_gaze_x'])
                # plt.plot(gazedata['left_pupil_measure1'])
                if tmin == 'et_looking_Start' and tmax == 'et_trial_End':
                    plt.axvline(x=etData.loc[etData['time']==trials.loc[index,'et_decision_Start']].index.values[0], color='r')
                    plt.axvline(x=etData.loc[etData['time']==trials.loc[index,'et_LaN_Start']].index.values[0], color='r')
                # plt.title('LaN = '+str(trial['LaN']))
                plt.axhline(y=gaze_threshold, color='g',linestyle='--')
                plt.axhline(y=1-gaze_threshold, color='g',linestyle='--')
                plt.axhline(y=-gaze_threshold, color='g',linestyle='--')
                plt.axhline(y=-1+gaze_threshold, color='g',linestyle='--')
                plt.show()   
    # def plot_pupil_dilation(self):
    #     etData = self.etData
        
    #     gaze_threshold = self.gaze_threshold
    #     trials = self.trials

    #     if self.task == 'gambling_hospital':
    #         time_slots = [{'tmin':'et_both_offers_Start','tmax':'et_LaN_Start', 'phase': 'stimuli presentation'},
    #                       {'tmin':'et_LaN_Start','tmax':'et_decision_Start', 'phase': 'LaN phase'}]
    #     else:
    #         time_slots = [{'tmin':self.tmin,'tmax':'et_LaN_Start', 'phase': 'stimuli presentation'},
    #                       {'tmin':'et_LaN_Start','tmax':'et_decision_Start', 'phase': 'LaN/LaS phase'}]
    #     for time_slot in time_slots:

    #         trials_LaN = trials.drop(trials[trials['LaN']==1].index).reset_index(drop=True)
    #         trials_LaS = trials.drop(trials[trials['LaN']==0].index).reset_index(drop=True)
    #         pupil_mean = []
    #         for index, trial in trials_LaN.iterrows():
               
    #             # select eyetracking data for the trial:  
    #             gazedata = select_eyedata(trial,index,etData,time_slot['tmin'],time_slot['tmax'],gaze_threshold)
                
    #             pupil_mean.append(np.mean(gazedata['left_pupil_measure1']))
 
    #         trials['pupil_mean'] = pupil_mean
        
        
        
            
    def plot_fraction_of_looking_time(self, LaN = None, LaS = None, chosen_option=None, distribution=None):
        '''
        plots the fraction of looking time for all trials in the same graph

        '''
        etData = self.etData
        
        gaze_threshold = self.gaze_threshold
        
        # time slots considering:
            # 1. Just the stimulus presentation.
            # 2. Just the 'real' observation phase, between stimulus presentation and sound presentation (LaN/LaS in food and context, LaN in gambling)
        
        if self.task == 'gambling_hospital':
            time_slots = [{'tmin':'et_both_offers_Start','tmax':'et_LaN_Start', 'phase': 'stimuli presentation'},
                          {'tmin':'et_LaN_Start','tmax':'et_decision_Start', 'phase': 'LaN phase'}]
        else:
            time_slots = [{'tmin':self.tmin,'tmax':'et_LaN_Start', 'phase': 'stimuli presentation'},
                          {'tmin':'et_LaN_Start','tmax':'et_decision_Start', 'phase': 'LaN/LaS phase'}]
        
        for time_slot in time_slots:
            
            trials = self.trials
            
            R_list = []
            L_list = []
        
            if LaN or LaS:
                if LaN:
                    trials = trials.drop(trials[trials['LaN']==1].index).reset_index(drop=True)
                if LaS:
                    trials = trials.drop(trials[trials['LaN']==0].index).reset_index(drop=True)
            
            if chosen_option:
                if chosen_option == 'right':
                    trials = trials.drop(trials[trials['response']==False].index).reset_index(drop=True)
                
                if chosen_option == 'left':
                    trials = trials.drop(trials[trials['response']==True].index).reset_index(drop=True)
        
        
            for index, trial in trials.iterrows():
               
                # select eyetracking data for the trial:  
                gazedata = select_eyedata(trial,index,etData,time_slot['tmin'],time_slot['tmax'],gaze_threshold)
                
                R_fraction, L_fraction = fraction_of_looking_time(gazedata, gaze_threshold,self.task)
                R_list.append(R_fraction)
                L_list.append(L_fraction)
                
            trials['R_fraction'] = R_list
            trials['L_fraction'] = L_list
            
            # Fit curve to fraction of looking times
            valuesRL = sorted(list(dict.fromkeys(trials['evidence_right'])), key = lambda x:float(x))
            ratiosR = []
            total_number_trials = []
            for valueRL in valuesRL:
                ratioR = sum(trials[trials['evidence_right'] == valueRL].R_fraction)/sum(trials['evidence_right'] == valueRL)
                total_number_trials.append(sum(trials['evidence_right'] == valueRL))
                ratiosR.append(ratioR)
            
            x = valuesRL
            y = ratiosR
            
            try:
                popt, pcov = opt.curve_fit(f, x, y, method="trf")
                
                fig, ax = plt.subplots(1, 1, figsize=(6, 4))
                ax.plot(x, y, 'ko', label='Data points')
                x = np.linspace(-1,1,50)
                y_fit = f(x, *popt)
                ax.plot(x, y_fit, 'k-',label='Logistic fit')
                plt.title('Fraction of Looking time (R) for ' + time_slot['phase'] + ' Subject: ' + str(self.subj))
                plt.legend()
                plt.xlim([-1,1])
                plt.ylim([0,1])
                plt.show()  
                
            except:
                x = valuesRL
                z = 1.96
                conf_int = [[y[i]-z*(np.sqrt((y[i]*(1-y[i]))/total_number_trials[i])) for i in range(len(total_number_trials))],
                            [y[i]+z*(np.sqrt((y[i]*(1-y[i]))/total_number_trials[i])) for i in range(len(total_number_trials))]]
                conf_int = [[y[i]-conf_int[0][i] for i in range(len(y))],[conf_int[1][i]-y[i] for i in range(len(y))]]
                plt.errorbar(valuesRL,y,conf_int,marker='o',color = 'k',label='Data points')
                plt.title('Fraction of Looking time (R) for ' + time_slot['phase'] + ' Subject: ' + str(self.subj))
                plt.legend()
                plt.xlim([-1,1])
                plt.ylim([0,1])
                plt.show()
            
            # Boxplots
            if distribution:
                boxplot = sns.boxplot(x='response',y='R_fraction',data=trials)
                boxplot.set(xlabel='Response (Chosen option)',ylabel='Fraction of right looking time ' + r'$t_R/(t_R+t_L)$')
                boxplot.set_xticklabels(['Left', 'Right'])
                boxplot.set(title='Distribution of Fraction of looking time for ' + time_slot['phase'] + ' Subject: ' + str(self.subj))
                plt.show()
                
    def decision_time(self):    
        return({'mean_decision_time': self.mean_decision_time, 'std_mean_decision_time': self.std_mean_decision_time,'max_decision_time':self.max_decision_time})
    
    def time_resolved(self, window_size=0.5, sliding=None, LaN=None, LaS=None):
        
        print('Time-resolved analysis Subject:'+str(self.subj)+'\n')
        
        # Param. initialization
        etData = self.etData
        gaze_threshold = self.gaze_threshold
        trials = self.trials
        
        # Condition to just select the LaN or the LaS trials
        if LaN or LaS:
            if LaN:
                trials = trials.drop(trials[trials['LaN']==1].index).reset_index(drop=True)
            if LaS:
                trials = trials.drop(trials[trials['LaN']==0].index).reset_index(drop=True)
        
        # Partition data into trials of right and left reported choice, respectively
        right_trials = trials.drop(trials[trials['response']==False].index).reset_index(drop=True)
        left_trials = trials.drop(trials[trials['response']==True].index).reset_index(drop=True)
        trials_divided = [right_trials,left_trials]
        
        # Define the time intervals depending on the task performed
        if self.task == 'gambling_hospital':
            time_slots = {'tmin':'et_both_offers_Start','tmax':'et_decision_Start'}
        else:
            time_slots = {'tmin':self.tmin,'tmax':'et_decision_Start'}
        
        # Initialize output 
        output = {}
        
        for chosen,trials in enumerate(trials_divided):
            
            # Initialize variables to store the fraction of looking time on each type of trial (chosen right or left)
            Chosen_lookingtime_R = []
            Chosen_lookingtime_L = []
            
            transitions_t = []
            
            for index, trial in trials.iterrows():
                
                # Create time windows
                windows = np.arange(trial[time_slots['tmin']], trial[time_slots['tmax']]+window_size, window_size)
                
                # Initialize variables to store the fraction of looking time on each window
                trial_lookingtime_R = []
                trial_lookingtime_L =[]
                
                transitions_time = []
                
                for time in windows:
                    
                    # Select eye tracking data in the time comprehended inside each window
                    if sliding:
                        gazedata = select_eyedata(trial,index,etData,time,time+window_size,gaze_threshold)
                    else:
                        gazedata = select_eyedata(trial,index,etData,trial[time_slots['tmin']],time+window_size,gaze_threshold)
                    
                    # Store in lists the fraction of looking time during each time window on each trial
                    R_fraction, L_fraction  = fraction_of_looking_time(gazedata, gaze_threshold, self.task)
                    trial_lookingtime_R.append(R_fraction)
                    trial_lookingtime_L.append(L_fraction)
                    
                    
                    data_bool = [~np.isnan(i) for i in gazedata.position]
                    gazedata = gazedata[np.array(data_bool)]
                    try:
                        transition_count = (np.diff(gazedata['position'])!=0).sum()
                    except:
                        transition_count = 0
                    transitions_time.append(transition_count)

                # List with all the looking times per window for all trials
                Chosen_lookingtime_R.append(trial_lookingtime_R)
                Chosen_lookingtime_L.append(trial_lookingtime_L)
                transitions_t.append(transitions_time)
                

                    
                
                
                
            
            # Expand eye tracker data in cases where some signal is missed at the end (looking at the mouse
            maxlenR = max(set([len (item) for item in Chosen_lookingtime_R]))
            maxlenL = max(set([len (item) for item in Chosen_lookingtime_L]))
            maxlenT = max(set([len (item) for item in transitions_t]))
            maxlen = max([maxlenR,maxlenL])
            for i,item in enumerate(Chosen_lookingtime_R):
                if len(item)<maxlen:
                    Chosen_lookingtime_R[i] = item + [float('nan')]*(maxlen-len(item))
            for i,item in enumerate(Chosen_lookingtime_L):
                if len(item)<maxlen:
                    Chosen_lookingtime_L[i] = item + [float('nan')]*(maxlen-len(item))
            for i, item in enumerate(transitions_t):
                if len(item)<maxlenT:
                    transitions_t[i] = item + [float('nan')]*(maxlenT-len(item))
            
            # Mean and SEM of the fraction of looking times
            Chosen_lookingtime_R_mean = np.nanmean(Chosen_lookingtime_R, axis=0)
            Chosen_lookingtime_R_sem = np.nanstd(Chosen_lookingtime_R, axis=0)/np.sqrt(len(Chosen_lookingtime_R))
            Chosen_lookingtime_L_mean = np.nanmean(Chosen_lookingtime_L, axis=0)
            Chosen_lookingtime_L_sem = np.nanstd(Chosen_lookingtime_L, axis=0)/np.sqrt(len(Chosen_lookingtime_L))
            
            transitions_mean = np.nanmean(transitions_t,axis = 0)
            transitions_sem = np.nanstd(transitions_t,axis=0)/np.sqrt(len(transitions_t))
            
            # Split the distributions according to the phases
            middle = int(len(Chosen_lookingtime_R_mean)/2)
            first_half_R_mean = Chosen_lookingtime_R_mean[:middle]
            second_half_R_mean = Chosen_lookingtime_R_mean[middle:]
            first_half_L_mean = Chosen_lookingtime_L_mean[:middle]
            second_half_L_mean = Chosen_lookingtime_L_mean[middle:]
            
            # T-test
            S_t_test_stat, S_t_test_pval = sp.ttest_ind([x for x in first_half_R_mean if np.isnan(x) == False] , [x for x in first_half_L_mean if np.isnan(x) == False])
            O_t_test_stat, O_t_test_pval = sp.ttest_ind([x for x in second_half_R_mean if np.isnan(x) == False] , [x for x in second_half_L_mean if np.isnan(x) == False])
            
            # Plot the resutls
            fig= plt.figure()
            plt.plot(Chosen_lookingtime_L_mean, label='L')
            plt.plot(Chosen_lookingtime_R_mean, label='R')
            plt.fill_between(list(range(len(Chosen_lookingtime_L_mean))), Chosen_lookingtime_L_mean+Chosen_lookingtime_L_sem, Chosen_lookingtime_L_mean-Chosen_lookingtime_L_sem,alpha=0.5)
            plt.fill_between(list(range(len(Chosen_lookingtime_R_mean))), Chosen_lookingtime_R_mean+Chosen_lookingtime_R_sem, Chosen_lookingtime_R_mean-Chosen_lookingtime_R_sem,alpha=0.5)
            fig.text(0.1, 0.1, r'$p-val = $'+str(round(S_t_test_pval,4)), fontsize = 12)
            fig.text(0.7, 0.1, r'$p-val = $'+str(round(O_t_test_pval,4)), fontsize = 12)
            fig.text(0.1, 0.9, 'Stimulation', fontsize = 12)
            fig.text(0.7, 0.9, 'Observation', fontsize = 12)           
            
            
            if LaN:
                plt.text(7, 0.8, 'LaN', fontsize = 12)
            elif LaS:
                plt.text(7, 0.8, 'LaS', fontsize = 12)
            
            if self.task == 'gambling_hospital':
                time_ticks = np.arange(0, 4+window_size, window_size)
                plt.xticks(ticks= list(range(len(windows))), labels=time_ticks)
                plt.axvline(x=2/window_size, color='g', alpha=0.5)
            if self.task == 'context':
                time_ticks = np.arange(0, 8+window_size, window_size)
                plt.xticks(ticks= list(range(len(windows))), labels=time_ticks)
                plt.axvline(x=3/window_size, color='g', alpha=0.5)
            if self.task == 'food':
                time_ticks = np.arange(0, 6+window_size, window_size)
                plt.xticks(ticks= list(range(len(windows))), labels=time_ticks)
                plt.axvline(x=3/window_size, color='g', alpha=0.5)
            # if self.task == 'gambling':
            #     time_ticks = np.arange(0, 6+window_size, window_size)
            #     plt.xticks(ticks= list(range(len(windows))), labels=time_ticks)
            #     plt.axvline(x=3/window_size, color='g', alpha=0.5)
    
            plt.xlim(0,len(windows)-1)
            plt.ylim(0,1)
            if chosen == 0:
                chosen_side = 'RIGHT'
            else:
                chosen_side = 'LEFT'
            plt.title('Time-resolved fraction of looking time for '+ chosen_side +' chosen option. \nTask: '+ str(self.task) + '. Subject:'+str(self.subj))
            plt.xlabel('Time [s]')
            plt.ylabel('Fraction of right looking time ' + r'$t/(t_R+t_L)$')
            plt.grid(visible=True, alpha=0.5)
            plt.legend()
            plt.savefig('results/'+str(self.task)+'/Time_resolved_'+chosen_side+'_chosen_TASK_'+str(self.task)+'_Subject_'+str(self.subj)+'.jpg', dpi=300)
            plt.show()
            
            if chosen == 0:
                output['Chosen_R'] = {'looking_time_L':Chosen_lookingtime_L_mean,'looking_time_R':Chosen_lookingtime_R_mean}
            else:
                output['Chosen_L'] = {'looking_time_L':Chosen_lookingtime_L_mean,'looking_time_R':Chosen_lookingtime_R_mean}
        
        
        # Plot the resutls
        fig = plt.figure()
        plt.plot(transitions_mean, label='transitions')
        plt.fill_between(list(range(len(transitions_mean))), transitions_mean+transitions_sem, transitions_mean-transitions_sem,alpha=0.5)
        # plt.text(1, 0.1, r'$p-val = $'+str(round(S_t_test_pval,4)), fontsize = 12)
        # plt.text(7, 0.1, r'$p-val = $'+str(round(O_t_test_pval,4)), fontsize = 12)
        fig.text(0.1, 0.9, 'Stimulation', fontsize = 12)
        fig.text(0.7, 0.9, 'Observation', fontsize = 12)           
        
        
        if LaN:
            plt.text(7, 0.8, 'LaN', fontsize = 12)
        elif LaS:
            plt.text(7, 0.8, 'LaS', fontsize = 12)
        
        if self.task == 'gambling_hospital':
            time_ticks = np.arange(0, 4+window_size, window_size)
            plt.xticks(ticks= list(range(len(windows))), labels=time_ticks)
            plt.axvline(x=2/window_size, color='g', alpha=0.5)
        if self.task == 'context':
            time_ticks = np.arange(0, 8+window_size, window_size)
            plt.xticks(ticks= list(range(len(windows))), labels=time_ticks)
            plt.axvline(x=3/window_size, color='g', alpha=0.5)
        if self.task == 'food':
            time_ticks = np.arange(0, 6+window_size, window_size)
            plt.xticks(ticks= list(range(len(windows))), labels=time_ticks)
            plt.axvline(x=3/window_size, color='g', alpha=0.5)
        # if self.task == 'gambling':
        #     time_ticks = np.arange(0, 6+window_size, window_size)
        #     plt.xticks(ticks= list(range(len(windows))), labels=time_ticks)
        #     plt.axvline(x=3/window_size, color='g', alpha=0.5)

        plt.xlim(0,len(windows)-1)
        # plt.ylim(0,1)
        if chosen == 0:
            chosen_side = 'RIGHT'
        else:
            chosen_side = 'LEFT'
        plt.title('Time-resolved transitions count \nTask: '+ str(self.task) + '. Subject:'+str(self.subj))
        plt.xlabel('Time [s]')
        plt.ylabel('n transitions')
        plt.grid(visible=True, alpha=0.5)
        plt.legend()
        # plt.savefig('results/'+str(self.task)+'/Time_resolved_'+chosen_side+'_chosen_TASK_'+str(self.task)+'_Subject_'+str(self.subj)+'.jpg', dpi=300)
        plt.show()
        
        return(output)
        
    def looking_time_fit(self,LaN=None,LaS=None):
        
        print('\nComputing logistic regression according to looking time of Subject:'+str(self.subj)+'...')
        
        etData = self.etData
        gaze_threshold = self.gaze_threshold
        
        if self.task == 'gambling_hospital':
            time_slots = [{'tmin':'et_both_offers_Start','tmax':'et_LaN_Start', 'phase': 'stimuli presentation'},
                          {'tmin':'et_LaN_Start','tmax':'et_decision_Start', 'phase': 'LaN phase'},
                          {'tmin':'et_both_offers_Start','tmax':'et_decision_Start', 'phase': 'pres+obv'},
                          {'tmin':'et_decision_Start','tmax':self.tmax, 'phase': 'decision'},
                          {'tmin':'et_both_offers_Start','tmax':self.tmax, 'phase': 'whole trial'}]
        else:
            time_slots = [{'tmin':self.tmin,'tmax':'et_LaN_Start', 'phase': 'stimuli presentation'},
                          {'tmin':'et_LaN_Start','tmax':'et_decision_Start', 'phase': 'LaN/LaS phase'},
                          {'tmin':self.tmin,'tmax':'et_decision_Start', 'phase': 'pres+obv'},
                          {'tmin':'et_decision_Start','tmax':self.tmax, 'phase': 'decision'},
                          {'tmin':self.tmin,'tmax':self.tmax, 'phase': 'whole trial'}]
        
        for time_slot in time_slots:
            
            trials = self.trials
            
            if LaN or LaS:
                if LaN:
                    if self.task == 'gambling_hospital':
                        time_slot = {'tmin':'et_LaN_Start','tmax':'et_decision_Start', 'phase': 'LaN phase'}
                    else:
                        trials = trials.drop(trials[trials['LaN']==1].index).reset_index(drop=True)
                if LaS:
                    if self.task == 'gambling_hospital':
                        time_slot = {'tmin':'et_both_offers_Start','tmax':'et_LaN_Start', 'phase': 'stimuli presentation'}
                    else:
                        trials = trials.drop(trials[trials['LaN']==0].index).reset_index(drop=True)
            
            R_list = []
            L_list = []
            
            for index, trial in trials.iterrows():
               
                # select eyetracking data for the trial:  
                gazedata = select_eyedata(trial,index,etData,time_slot['tmin'],time_slot['tmax'],gaze_threshold)
                
                R_fraction, L_fraction = fraction_of_looking_time(gazedata, gaze_threshold,self.task)
                R_list.append(R_fraction)
                L_list.append(L_fraction)
            
            trials['n_response'] = trials['response']*1
            trials['R_fraction'] = R_list
            trials['L_fraction'] = L_list
            
            try: 
                mod = smf.glm(formula= 'n_response ~ R_fraction' , data= trials , family= sm.families.Binomial() )
                res = mod.fit()
        
                plt.figure(figsize=(6, 6), dpi=300)
                
                Y= trials.n_response 
                X1=np.round(trials.R_fraction,1)
                nx = np.unique(X1)
                mn=[np.mean(Y[X1==x]) for x in nx]
                sem=[sp.sem(Y[X1==x]) for x in nx] 
                
                plt.plot(nx,mn, color='slategrey',label='Mean')
                plt.fill_between(nx, np.array(mn)+np.array(sem), np.array(mn)-np.array(sem),alpha=0.5,color='lightsteelblue',label='SEM')
    
                myx = np.linspace(0,1,100)
                line_fit=res.predict(pd.DataFrame({'R_fraction': myx}))
                plt.plot(myx,line_fit,'-',color='k', linewidth=3,label='Log Fit')
                
                plt.text(0.6, 0.35, r'$\beta = $'+str(round(res.params.R_fraction,3)), fontsize = 12)
                plt.text(0.6, 0.3, r'$p-value = $'+str(round(res.pvalues.R_fraction,3)), fontsize = 12)
                
                plt.xlim((0,1))
                plt.ylim((0,1))
                plt.grid(visible=True)
                plt.title('\n'.join(wrap('Probability of reporting the right choice according to fraction of looking time to the right option.'+' Task: '+self.task+'; Phase: '+time_slot['phase']+'; S:'+str(self.subj),60)), fontsize=12)
    
                plt.xlabel('Fraction of looking time [R]')
                plt.ylabel('Probability of reporting right choice')
                plt.legend()
                plt.show()
            
            except:
                print('Perfect separation in Phase: '+time_slot['phase'])
        
        
        return(list(trials['n_response']), list(trials['R_fraction']))




### VARIABLES ###
# Directory with the data
fileDir_food = r"data_food/"
fileDir_context = r"data_context/"
fileDir_gamblingH = r"data_gambling_HOSPITAL/"

fileDir = [fileDir_food,fileDir_gamblingH,fileDir_context]
tasks = ['food','gambling']

subjects_food = [1]
subjects_gamblingH = [1,2,3,4]
subjects_context = [1,2,4]

subjects = [subjects_food,subjects_gamblingH,subjects_context]

# ### RUN CODE ###

# # ### FOR ALL TASKS AND ALL SUBJECTS
# for t in range(len(tasks)):
#     print('Computing '+tasks[t]+' task')
#     trials_all = pd.DataFrame(columns = trials.keys())
#     for subj in subjects[t]:
#         print('Computing subject ',subj)
#         subject_n = subject(fileDir[t],subj,tmin ='et_looking_Start', tmax = 'et_decision_Start')
#         trials_all = trials_all.append(subject_n.trials)
#         # subject_1.transitions()
#         # subject_1.psychometric_curve()

#     plot_corr(trials_all,'total_value','confidence',tasks[t])
#     plot_corr(trials_all,'total_value','transitions',tasks[t])
#     plot_corr(trials_all,'difficulty','confidence',tasks[t])
#     plot_corr(trials_all,'difficulty','transitions',tasks[t])
#     plot_corr(trials_all,'confidence','transitions',tasks[t])

### FOR A PARTICULAR TASK AND ALL SUBJECTS
t = 2
if t == 0 or t == 2:
    gaze_threshold = 0.2
else:
    gaze_threshold = 0.2
trials_all = pd.DataFrame()
for subj in subjects[t]:
    print('Computing subject ',subj)
    print(fileDir)
    
    if subj == 1:
        subject_n = subject(fileDir[t],subj,tmin ='et_looking_Start', tmax = 'et_decision_Start',gaze_threshold=gaze_threshold)
        print(len(trials))
        trials_all = pd.DataFrame(columns = trials.keys())
        trials_all = trials_all.append(subject_n.trials)
        subject_n.psycho_justo()
        # if t == 0 or t == 2:
        #     subject_n.heatmap(section='presentation',chosen_option='left',LaN=True)
        #     subject_n.heatmap(section='presentation',chosen_option='right',LaN=True)
        #     subject_n.heatmap(section='observation',chosen_option='left',LaN=True)
        #     subject_n.heatmap(section='observation',chosen_option='right',LaN=True)
        #     subject_n.heatmap(section='decision',chosen_option='left',LaN=True)
        #     subject_n.heatmap(section='decision',chosen_option='right',LaN=True)
        # subject_n.time_resolved()
    else:
        subject_n = subject(fileDir[t],subj,tmin ='et_looking_Start', tmax = 'et_decision_Start',gaze_threshold=gaze_threshold)
        print(len(trials))
        trials_all = trials_all.append(subject_n.trials)
        # subject_1.transitions()
        subject_n.psycho_justo()
        # subject_n.time_resolved()

# plot_corr(trials_all,'total_value','confidence',tasks[t])
# plot_corr(trials_all,'total_value','transitions',tasks[t])
# plot_corr(trials_all,'difficulty','confidence',tasks[t])
# plot_corr(trials_all,'difficulty','transitions',tasks[t])
# plot_corr(trials_all,'confidence','transitions',tasks[t])
# plot_corr(trials_all,'total_value','consistency',tasks[t])
# plot_corr(trials_all,'transitions','confidence',tasks[t])
# plot_corr(trials_all,'chosen_lt','confidence',tasks[t])
# plot_corr(trials_all,'right_lt','response',tasks[t])

# if t==0:
#     plot_corr(trials_all,'confidence','consistency',tasks[t])





### VARIABLES ###
# # Directory with the data
# fileDir_food = r"data_pilot_food/"
# fileDir_gambling = r"data_pilot_gambling_HEALTHY/"
# fileDir_context = r"data_pilot_context/"
# fileDir_gamblingH = r"data_pilot_gambling_HOSPITAL/"

# fileDir = [fileDir_food,fileDir_context,fileDir_gamblingH]
# tasks = ['food','context','gambling']

# subjects_food = [7,8]
# subjects_gamblingH = [10,11,12]
# subjects_context = [1,2,3,4]
# subjects = [subjects_food,subjects_context,subjects_gamblingH]

### RUN CODE ###

### FOR ALL TASKS AND ALL SUBJECTS
# for t in range(len(tasks)):
#     print('Computing '+tasks[t]+' task')
#     trials_all = pd.DataFrame()
#     for subj in subjects[t]:
#         print('Computing subject ',subj)
#         subject_n = subject(fileDir[t],subj,tmin ='et_looking_Start', tmax = 'et_decision_Start')
#         trials_all = trials_all.append(subject_n.trials)
#         # subject_1.transitions()
#         # subject_1.psychometric_curve()

#     plot_corr(trials_all,'total_value','confidence',tasks[t])
#     plot_corr(trials_all,'total_value','transitions',tasks[t])
#     plot_corr(trials_all,'difficulty','confidence',tasks[t])
#     plot_corr(trials_all,'difficulty','transitions',tasks[t])
#     plot_corr(trials_all,'confidence','transitions',tasks[t])




# subj = 1

# subject = subject(fileDir_gambling_hospital,subj, gaze_threshold=0)
# subject.looking_time_fit()

# subject.time_resolved(window_size=0.5)

# subject.heatmap(section='right',chosen_option='left')
# subject.heatmap(section='left',chosen_option='left')
# subject.heatmap(section='both',chosen_option='left')
# subject.heatmap(section='observation',chosen_option='left')
# subject.heatmap(section='decision',chosen_option='left')

# subject.heatmaps(confidence=True)
# subject.heatmaps(difficulty=True)




### RUN CODE ###

# # # CONTEXT
# for subj in [1,2,3,4,5]:
# # for subj in [5]:
#     sub = subject(fileDir_context,subj, gaze_threshold=0.375)
#     sub.looking_time_fit()
#     subject_1.time_resolved(window_size=0.5)
    # subject_1.heatmaps(confidence=True)
    # subject_1.heatmap(section='presentation',chosen_option='left')
    # subject_1.heatmap(section='presentation',chosen_option='right')
    # subject_1.heatmap(section='observation',chosen_option='left')
    # subject_1.heatmap(section='observation',chosen_option='right')
    # subject_1.heatmap(section='decision',chosen_option='left')
    # subject_1.heatmap(section='decision',chosen_option='right')

    

# # FOOD
# for subj in [7,8]:
#     subject_1 = subject(fileDir_food,subj, gaze_threshold=0.375)
#     subject_1.time_resolved(window_size=0.5)
#     subject_1.heatmaps(confidence=True)
#     subject_1.heatmap(section='observation',chosen_option='right')
#     # subject_1.plot_eyedata_trials()
#     # subject_1.plot_fraction_of_looking_time(distribution=True)

# # GAMBLING HOSPITAL 











# responses = []
# Rs = []

# # # for subj in [5]:

    


# TR_subs =[]
# # # for subj in [7,8]:
# for subj in [1,2,3]:

#     sub = subject(fileDir_gambling,subj, gaze_threshold=0.1)
#     tr = sub.time_resolved(window_size=0.5,LaN=True,sliding = True)
#     # tr = sub.time_resolved(window_size=0.5,LaS=True,sliding = True)
#     # tr = sub.time_resolved(window_size=0.5,sliding = True)
#     TR_subs.append(tr)
#     # tr = sub.time_resolved(window_size=0.5,sliding=True,LaN=True)
#     # tr = sub.time_resolved(window_size=0.5,sliding=True,LaS=True)
#     # plt.imshow([[0,0],[0,0]])
 

    
# lRcR = []
# lLcR = []
# for sub in TR_subs:
#     lRcR.append(sub['Chosen_R']['looking_time_R'])
#     lLcR.append(sub['Chosen_R']['looking_time_L'])
    
# lRcR_mean = np.nanmean(lRcR,axis=0)
# lLcR_mean = np.nanmean(lLcR,axis=0)


# # Mean and SEM of the fraction of looking times
# lRcR_sem = np.nanstd(lRcR_mean, axis=0)/np.sqrt(len(lRcR_mean))
# lLcR_sem = np.nanstd(lLcR_mean, axis=0)/np.sqrt(len(lLcR_mean))

# # Split the distributions according to the phases
# middle = int(len(lRcR_mean)/2)
# first_half_R_mean = lRcR_mean[:middle]
# second_half_R_mean = lRcR_mean[middle:]
# first_half_L_mean = lRcR_mean[:middle]
# second_half_L_mean = lRcR_mean[middle:]

# # T-test
# S_t_test_stat, S_t_test_pval = sp.ttest_ind([x for x in first_half_R_mean if np.isnan(x) == False] , [x for x in first_half_L_mean if np.isnan(x) == False])
# O_t_test_stat, O_t_test_pval = sp.ttest_ind([x for x in second_half_R_mean if np.isnan(x) == False] , [x for x in second_half_L_mean if np.isnan(x) == False])


# # plt.fill_between(list(range(len(Chosen_lookingtime_L_mean))), Chosen_lookingtime_L_mean+Chosen_lookingtime_L_sem, Chosen_lookingtime_L_mean-Chosen_lookingtime_L_sem,alpha=0.5)
# # plt.fill_between(list(range(len(Chosen_lookingtime_R_mean))), Chosen_lookingtime_R_mean+Chosen_lookingtime_R_sem, Chosen_lookingtime_R_mean-Chosen_lookingtime_R_sem,alpha=0.5)
# fig= plt.figure(dpi=300)
# plt.plot(lLcR_mean,label='L')
# plt.plot(lRcR_mean,label='R')
# plt.fill_between(list(range(len(lLcR_mean))), lLcR_mean+lLcR_sem, lLcR_mean-lLcR_sem,alpha=0.5)
# plt.fill_between(list(range(len(lRcR_mean))), lRcR_mean+lRcR_sem, lRcR_mean-lRcR_sem,alpha=0.5)
# # plt.text(7, 0.8, 'LaN', fontsize = 12)
# fig.text(0.1, 0.9, 'Stimulation', fontsize = 12)
# fig.text(0.7, 0.9, 'Observation', fontsize = 12)

# time_ticks = np.arange(0, 5.5, 0.5)
# plt.xticks(ticks= list(range(len(time_ticks))), labels=time_ticks)
# plt.axvline(x=3/0.5, color='g', alpha=0.5)

# plt.xlim((0,10))
# plt.ylim((0,1))

# plt.grid()

# plt.xlabel('Time [s]')
# plt.ylabel('Fraction of looking time')

# plt.legend()
# plt.show()














# looking_times_R = []
# looking_times_L = []
# for index, trial in trials.iterrows():
    
#     gazedata = select_eyedata(trial, index,etData, 'et_LaN_Start', 'et_decision_Start',gaze_threshold=0.375)
#     lt_R, lt_L = fraction_of_looking_time(gazedata,0.375,'food')
#     looking_times_R.append(lt_R)
#     looking_times_L.append(lt_L)
    
# trials['LT_R'] =  looking_times_R
# trials['LT_L'] =  looking_times_L

# looking_times = []
# axis = []
# for value in range(-10,11):
    
#     lt = (trials[trials.stimL_value == value].LT_L.mean() + trials[trials.stimR_value == value].LT_R.mean())/2
#     looking_times.append(lt)
#     axis.append(value)
    
    
    
    
    
    
    
    
# for subj in [7,8]:
    
#     sub = subject(fileDir_food,subj, gaze_threshold=0.375)
#     resp, R_frac = sub.looking_time_fit()
    
#     try:
#         responses = responses + resp
#         Rs = Rs + R_frac
#     except:
#         responses = resp
#         Rs = R_frac
        
    
# subs_RF = pd.DataFrame({'n_response':responses,
#               'R_fraction':Rs})
# subs_RF = subs_RF.dropna(axis=0).reset_index()

# mod = smf.glm(formula= 'n_response ~ R_fraction' , data= subs_RF , family= sm.families.Binomial() )
# res = mod.fit()

# prediction = res.predict(subs_RF.R_fraction)
# labelled_prediction = [int(x>=0.5) for x in prediction]
# accuracy = round(np.sum(labelled_prediction == subs_RF.n_response)/len(subs_RF),3)
# plt.plot(subs_RF.R_fraction,subs_RF.n_response,'o')

# plt.figure(figsize=(6, 6), dpi=300)

# Y= subs_RF.n_response 
# X1=np.round(subs_RF.R_fraction,1)
# nx = np.unique(X1)
# mn=[np.mean(Y[X1==x]) for x in nx]
# sem=[sp.sem(Y[X1==x]) for x in nx] 

# plt.plot(nx,mn, color='slategrey',label='Mean')
# # plt.plot(nx,mn, 'o',color='slategrey')
# plt.fill_between(nx, np.array(mn)+np.array(sem), np.array(mn)-np.array(sem),alpha=0.5,color='lightsteelblue',label='SEM')

# myx = np.linspace(0,1,100)
# line_fit=res.predict(pd.DataFrame({'R_fraction': myx}))
# plt.plot(myx,line_fit,'-',color='k', linewidth=3, label='Log Fit')

# # plt.errorbar(nx,mn,yerr=sem, fmt="o", capsize=4, color='slategrey')

# plt.text(0.6, 0.35, r'$\beta = $'+str(round(res.params.R_fraction,3)), fontsize = 12)
# plt.text(0.6, 0.3, r'$p-value = $'+str(round(res.pvalues.R_fraction,3)), fontsize = 12)

# plt.xlabel('Fraction of looking time [R]')
# plt.ylabel('Probability of reporting right choice')
# plt.title('\n'.join(wrap('Probability of reporting the right choice according to fraction of looking time to the right option',60)), fontsize=12)

# plt.xlim((0,1))
# plt.ylim((0,1))
# plt.legend()
# plt.grid(visible=True)






















#     subject_1.heatmap(section='right',chosen_option='left')
#     subject_1.heatmap(section='left',chosen_option='left')
#     subject_1.heatmap(section='both',chosen_option='left')
#     subject_1.heatmap(section='observation',chosen_option='left')
#     subject_1.heatmap(section='decision',chosen_option='left')
#     # subject_1.heatmaps(difficulty=True)
    # subject_1.heatmap(section='observation',chosen_option='right')
    
    # subject_1.plot_fraction_of_looking_time(distribution=True)


# subject_1.plot_fraction_of_looking_time(LaS=True, distribution=True)
# subject_1.plot_fraction_of_looking_time(tmin='et_looking_Start', tmax='et_LaN_Start')
# subject_1.plot_fraction_of_looking_time(tmin='et_LaN_Start', tmax='et_decision_Start', LaS=True)
# subject_1.plot_fraction_of_looking_time(tmin='et_decision_Start', tmax='et_trial_End', LaS=True)

# subject_1.plot_fraction_of_looking_time(LaN=True, distribution=True)
# subject_1.plot_fraction_of_looking_time(tmin='et_LaN_Start', tmax='et_decision_Start', LaN=True)
# subject_1.plot_fraction_of_looking_time(tmin='et_decision_Start', tmax='et_trial_End', LaN=True)

# subject_1.heatmap()

# subject_1.heatmap(section='presentation',chosen_option='right',LaN=True)
# subject_1.heatmap(section='observation',chosen_option='right',LaN=True)
# subject_1.heatmap(section='decision',chosen_option='right',LaN=True)

# subject_1.heatmap(section='presentation',chosen_option='left',LaN=True)
# subject_1.heatmap(section='observation',chosen_option='left',LaN=True)
# subject_1.heatmap(section='decision',chosen_option='left',LaN=True)

# subject_1.heatmap(section='presentation',chosen_option='right',LaS=True)
# subject_1.heatmap(section='observation',chosen_option='right',LaS=True)
# subject_1.heatmap(section='decision',chosen_option='right',LaS=True)

# subject_1.heatmap(section='presentation',chosen_option='left',LaS=True)
# subject_1.heatmap(section='observation',chosen_option='left',LaS=True)
# subject_1.heatmap(section='decision',chosen_option='left',LaS=True)


## DEBUG
# etData = subject_1.etData
# trials = subject.trials
# subject_1.heatmap(section='right')
# subject_1.heatmap(section='left')
# subject_1.heatmap(section='both')
# subject_1.heatmap(section='decision')

# subject_1.heatmap(section='right',chosen_option='right')
# subject_1.heatmap(section='left',chosen_option='right')
# subject_1.heatmap(section='both',chosen_option='right')
# subject_1.heatmap(section='decision',chosen_option='right')

# subject_1.heatmap(section='right',chosen_option='left')
# subject_1.heatmap(section='left',chosen_option='left')
# subject_1.heatmap(section='both',chosen_option='left')
# subject_1.heatmap(section='decision',chosen_option='left')

# subject_1.plot_fraction_of_looking_time()
# subject_1.plot_fraction_of_looking_time(tmin='et_both_offers_Start',tmax='et_decision_Start')

# subject_1.plot_fraction_of_looking_time(tmin='et_decision_Start', tmax = 'et_trial_End')





#     # Image presentation 
#     accuracy = analize(trials,etData,'et_looking_Start','et_decision_Start', heatmap=True, chosen_option='right')
#     accuracy = analize(trials,etData,'et_looking_Start','et_decision_Start', heatmap=True, chosen_option='left')
#     print('  STIM accuracy: ', accuracy)
    
#     # Image presentation + LaN / LaS
#     accuracy = analize(trials,etData,'et_looking_Start','et_trial_End')
#     print('  STIM + LaN/LaS accuracy: ', accuracy)
    
#     # Image presentation + LaN
#     accuracy = analize(trials,etData,'et_looking_Start','et_trial_End',LaN=True)
#     print('  STIM + LaN accuracy: ', accuracy)
    
#     # Image presentation + LaS
#     accuracy = analize(trials,etData,'et_looking_Start','et_trial_End',LaS=True)
#     print('  STIM + LaS accuracy: ', accuracy)
    
#     # LaN / LaS
#     accuracy = analize(trials,etData,'et_decision_Start','et_trial_End')
#     print('  LaN/LaS accuracy: ', accuracy)
    
#     # LaN
#     accuracy = analize(trials,etData,'et_decision_Start','et_trial_End', LaN=True, heatmap=True, chosen_option='right')
#     accuracy = analize(trials,etData,'et_decision_Start','et_trial_End', LaN=True, heatmap=True, chosen_option='left')
#     print('  LaN accuracy: ', accuracy)
    
#     # LaS
#     accuracy = analize(trials,etData,'et_decision_Start','et_trial_End', LaS=True, heatmap=True, chosen_option='right')
#     accuracy = analize(trials,etData,'et_decision_Start','et_trial_End', LaS=True, heatmap=True, chosen_option='left')
#     print('  LaS accuracy: ', accuracy, '\n')

