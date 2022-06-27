#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2021.2.3),
    on Sun Jun 26 21:02:55 2022
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

from __future__ import absolute_import, division

from psychopy import locale_setup
from psychopy import prefs
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, iohub, hardware
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

import numpy as np
#from pyfirmata import Arduino
#arduino_pin = (7)
#arduino_board = Arduino('/dev/cu.usbmodem11101')
#from pyfirmata import Arduino
#arduino_pin = (7)
#arduino_board = Arduino('/dev/cu.usbmodem11101')


# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Store info about the experiment session
psychopyVersion = '2021.2.3'
expName = 'gamblingtest11stversion '  # from the Builder filename that created this script
expInfo = {'participant': '', 'session': '001'}
dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopyVersion

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath='/Volumes/GoogleDrive/Mi unidad/_BRAIN+COGNITION/0_TFM/Experiments/Gambling_ESP/patientgambling.py',
    savePickle=True, saveWideText=True,
    dataFileName=filename)
# save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp
frameTolerance = 0.001  # how close to onset before 'same' frame

# Start Code - component code to be run after the window creation

# Setup the Window
win = visual.Window(
    size=[1440, 900], fullscr=True, screen=0, 
    winType='pyglet', allowGUI=False, allowStencil=False,
    monitor='testMonitor', color='grey', colorSpace='rgb',
    blendMode='avg', useFBO=True, 
    units='height')
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess

# Setup eyetracking
ioDevice = ioConfig = ioSession = ioServer = eyetracker = None

# create a default keyboard (e.g. to check for escape)
defaultKeyboard = keyboard.Keyboard()

# Initialize components for Routine "instruction_for_caliberation_"
instruction_for_caliberation_Clock = core.Clock()
text = visual.TextStim(win=win, name='text',
    text='  \n\nA continuación verá unos puntos moverse por la pantalla. Por favor, siga los puntos con la mirada en todo momento. \n\nHaga click en el ratón para continuar.',
    font='Open Sans',
    pos=(0.05, 0.05), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
mouse_4 = event.Mouse(win=win)
x, y = [None, None]
mouse_4.mouseClock = core.Clock()
win.mouseVisible = False

# Initialize components for Routine "mouse_hidder"
mouse_hidderClock = core.Clock()
win.mouseVisible = False

# Initialize components for Routine "instruction_"
instruction_Clock = core.Clock()
text_3 = visual.TextStim(win=win, name='text_3',
    text='En esta prueba aparecerán columnas con diferentes colores. Cada color tiene un valor específico:\n\ngris = 5 puntos\nazul = 7 puntos\nverde = 10 puntos\n\nHaga click en el ratón para continuar.',
    font='Open Sans',
    pos=(0.05, 0.05), height=0.05, wrapWidth=None, ori=0.2, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
mouse_2 = event.Mouse(win=win)
x, y = [None, None]
mouse_2.mouseClock = core.Clock()

# Initialize components for Routine "instructions21"
instructions21Clock = core.Clock()
text_11 = visual.TextStim(win=win, name='text_11',
    text='La altura de cada columna le indica la probabilidad de conseguir esos puntos.\n\n\n\n\n\n\n\nEn este ejemplo, usted tendría un 50% de posibilidades de ganar 10 puntos, y un 50% de posibilidades de ganar 0 puntos.\n\nHaga click en el ratón para continuar.',
    font='Open Sans',
    pos=(0.05, 0.05), height=0.05, wrapWidth=None, ori=0.2, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
mouse_5 = event.Mouse(win=win)
x, y = [None, None]
mouse_5.mouseClock = core.Clock()
examplecolumn = visual.Rect(
    win=win, name='examplecolumn',
    width=(0.15, 0.35)[0], height=(0.15, 0.35)[1],
    ori=0.0, pos=(0, 0.15),
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='dark grey',
    opacity=None, depth=-2.0, interpolate=True)
examplecolor = visual.Rect(
    win=win, name='examplecolor',
    width=[1.0, 1.0][0], height=[1.0, 1.0][1],
    ori=0.0, pos=[0,0],
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-3.0, interpolate=True)

# Initialize components for Routine "instruction_2"
instruction_2Clock = core.Clock()
text_6 = visual.TextStim(win=win, name='text_6',
    text='Despues de cada intento, se le informará de cuantos puntos ha conseguido en ese intento, así como de la puntuación que lleva acumulada. \n\nPor favor, haga sus elecciones prestando atención, su objetivo en este experimento es el de conseguir la mayor puntuación posible. \n\nHaga click en el ratón para continuar.',
    font='Open Sans',
    pos=(0.05, 0.05), height=0.05, wrapWidth=None, ori=0.2, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
mouse_3 = event.Mouse(win=win)
x, y = [None, None]
mouse_3.mouseClock = core.Clock()

# Initialize components for Routine "instruction3"
instruction3Clock = core.Clock()
text_12 = visual.TextStim(win=win, name='text_12',
    text='Solo podrá marcar que columna ha elegido despues de oír un sonido, para ello haga click en el botón derecho o izquierdo del ratón.\n\nSi decide escoger la columna derecha, haga click en el botón derecho. Si decide escoger la columna izquierda, haga click en el botón izquierdo. \n\nHaga click en el ratón para continuar.',
    font='Open Sans',
    pos=(0.05, 0.05), height=0.05, wrapWidth=None, ori=0.2, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
mouse_6 = event.Mouse(win=win)
x, y = [None, None]
mouse_6.mouseClock = core.Clock()

# Initialize components for Routine "instructions4"
instructions4Clock = core.Clock()
text_17 = visual.TextStim(win=win, name='text_17',
    text='Cada vez que vea la siguiente cruz:\n\n\n\n\n\n\n\nHa de fijar la vista en ella.\n\nHaga click en el ratón para continuar.',
    font='Open Sans',
    pos=(0.05, 0.05), height=0.05, wrapWidth=None, ori=0.2, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
mouse_10 = event.Mouse(win=win)
x, y = [None, None]
mouse_10.mouseClock = core.Clock()
fix_instructions = visual.ShapeStim(
    win=win, name='fix_instructions', vertices='cross',
    size=[1.0, 1.0],
    ori=1.0, pos=[0,0],
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-2.0, interpolate=True)

# Initialize components for Routine "starttraining"
starttrainingClock = core.Clock()
text_16 = visual.TextStim(win=win, name='text_16',
    text='Ahora realizará 10 intentos de prueba. La puntuación en estos 10 intentos no se tendrá en cuenta para la puntuación final, es solo un entrenamiento. \n\nSi tiene alguna duda, por favor pregunte al experimentador. \n\nHaga click en el ratón para continuar.',
    font='Open Sans',
    pos=(0.05, 0.05), height=0.05, wrapWidth=None, ori=0.2, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
mouse_9 = event.Mouse(win=win)
x, y = [None, None]
mouse_9.mouseClock = core.Clock()

# Initialize components for Routine "calc_2"
calc_2Clock = core.Clock()
value_trial=0
true_choice=0
color_value={"lightgrey":5 ,"blue":7,"green":10}


# Initialize components for Routine "trial"
trialClock = core.Clock()
fix = visual.ShapeStim(
    win=win, name='fix', vertices='cross',
    size=[1.0, 1.0],
    ori=1.0, pos=[0,0],
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=0.0, interpolate=True)
etRecord_2 = hardware.eyetracker.EyetrackerControl(
    server=ioServer,
    tracker=eyetracker
)

# Initialize components for Routine "offer1_training"
offer1_trainingClock = core.Clock()
rl1_2 = visual.Rect(
    win=win, name='rl1_2',
    width=(0.3, 0.5)[0], height=(0.3, 0.5)[1],
    ori=0.0, pos=(-0.4, 0),
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='dark grey',
    opacity=None, depth=0.0, interpolate=True)
cl1_2 = visual.Rect(
    win=win, name='cl1_2',
    width=[1.0, 1.0][0], height=[1.0, 1.0][1],
    ori=0.0, pos=[0,0],
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-1.0, interpolate=True)
etRecord_8 = hardware.eyetracker.EyetrackerControl(
    server=ioServer,
    tracker=eyetracker
)
text_13 = visual.TextStim(win=win, name='text_13',
    text='',
    font='Open Sans',
    pos=[0,0], height=0.1, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-3.0);

# Initialize components for Routine "delay1"
delay1Clock = core.Clock()
image_2 = visual.ImageStim(
    win=win,
    name='image_2', 
    image=None, mask=None,
    ori=0.0, pos=(0, 0), size=(0.5, 0.5),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=0.0)
fix_2 = visual.ShapeStim(
    win=win, name='fix_2', vertices='cross',
    size=[1.0, 1.0],
    ori=1.0, pos=[0,0],
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-1.0, interpolate=True)

# Initialize components for Routine "offer2_training"
offer2_trainingClock = core.Clock()
rr1_2 = visual.Rect(
    win=win, name='rr1_2',
    width=(0.3, 0.5)[0], height=(0.3, 0.5)[1],
    ori=0.0, pos=(0.4, 0),
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='dark grey',
    opacity=None, depth=0.0, interpolate=True)
cr1_2 = visual.Rect(
    win=win, name='cr1_2',
    width=[1.0, 1.0][0], height=[1.0, 1.0][1],
    ori=0.0, pos=[0,0],
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-1.0, interpolate=True)
etRecord_9 = hardware.eyetracker.EyetrackerControl(
    server=ioServer,
    tracker=eyetracker
)
text14 = visual.TextStim(win=win, name='text14',
    text='',
    font='Open Sans',
    pos=[0,0], height=0.1, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-3.0);

# Initialize components for Routine "delay2"
delay2Clock = core.Clock()
image_3 = visual.ImageStim(
    win=win,
    name='image_3', 
    image=None, mask=None,
    ori=0.0, pos=(0, 0), size=(0.5, 0.5),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=0.0)
fix_3 = visual.ShapeStim(
    win=win, name='fix_3', vertices='cross',
    size=[1.0, 1.0],
    ori=1.0, pos=[0,0],
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-1.0, interpolate=True)

# Initialize components for Routine "observ_training"
observ_trainingClock = core.Clock()
rl_2 = visual.Rect(
    win=win, name='rl_2',units='height', 
    width=[1.0, 1.0][0], height=[1.0, 1.0][1],
    ori=0.0, pos=[0,0],
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='dark grey',
    opacity=None, depth=0.0, interpolate=True)
cl_2 = visual.Rect(
    win=win, name='cl_2',units='height', 
    width=[1.0, 1.0][0], height=[1.0, 1.0][1],
    ori=0.0, pos=[0,0],
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-1.0, interpolate=True)
rr_2 = visual.Rect(
    win=win, name='rr_2',units='height', 
    width=[1.0, 1.0][0], height=[1.0, 1.0][1],
    ori=0.0, pos=[0,0],
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='dark grey',
    opacity=None, depth=-2.0, interpolate=True)
cr_2 = visual.Rect(
    win=win, name='cr_2',units='height', 
    width=[1.0, 1.0][0], height=[1.0, 1.0][1],
    ori=0.0, pos=[0,0],
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-3.0, interpolate=True)
etRecord_10 = hardware.eyetracker.EyetrackerControl(
    server=ioServer,
    tracker=eyetracker
)
text_15 = visual.TextStim(win=win, name='text_15',
    text='',
    font='Open Sans',
    pos=[0,0], height=0.1, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-5.0);
text14_3 = visual.TextStim(win=win, name='text14_3',
    text='',
    font='Open Sans',
    pos=[0,0], height=0.1, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-6.0);

# Initialize components for Routine "decision"
decisionClock = core.Clock()
decr = visual.Rect(
    win=win, name='decr',
    width=[1.0, 1.0][0], height=[1.0, 1.0][1],
    ori=1.0, pos=[0,0],
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor=None,
    opacity=None, depth=0.0, interpolate=True)
dec2 = visual.Rect(
    win=win, name='dec2',
    width=[1.0, 1.0][0], height=[1.0, 1.0][1],
    ori=1.0, pos=[0,0],
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor=None,
    opacity=None, depth=-1.0, interpolate=True)
etRecord_4 = hardware.eyetracker.EyetrackerControl(
    server=ioServer,
    tracker=eyetracker
)
sound_1 = sound.Sound('A', secs=0.5, stereo=True, hamming=True,
    name='sound_1')
sound_1.setVolume(1.0)
mouse = event.Mouse(win=win)
x, y = [None, None]
mouse.mouseClock = core.Clock()

# Initialize components for Routine "confidence"
confidenceClock = core.Clock()
slider = visual.Slider(win=win, name='slider',
    startValue=None, size=(1.0, 0.1), pos=(0, 0), units=None,
    labels=("-","+"), ticks=(1, 2, 3, 4, 5 , 6 ,7 ,8 , 9 ,10), granularity=1.0,
    style='radio', styleTweaks=(), opacity=None,
    color='white', fillColor='Red', borderColor='white', colorSpace='rgb',
    font='Open Sans', labelHeight=0.05,
    flip=False, depth=0, readOnly=False)
text_2 = visual.TextStim(win=win, name='text_2',
    text='¿Qué nivel de seguridad siente sobre la decisión que acaba de tomar?',
    font='Open Sans',
    pos=(0, 0.1), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-1.0);

# Initialize components for Routine "score"
scoreClock = core.Clock()
score1 = visual.TextStim(win=win, name='score1',
    text='',
    font='Open Sans',
    pos=(-0.1, 0.05), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
Score = visual.TextBox2(
     win, text='', font='Open Sans',
     pos=(0, 0.05),     letterHeight=0.02,
     size=(0.25,0.1), borderWidth=4.0,
     color='black', colorSpace='rgb',
     opacity=None,
     bold=False, italic=False,
     lineSpacing=1.0,
     padding=0.0,
     anchor='center',
     fillColor=None, borderColor='black',
     flipHoriz=False, flipVert=False,
     editable=False,
     name='Score',
     autoLog=True,
)
accumulatedvalue = visual.TextStim(win=win, name='accumulatedvalue',
    text='',
    font='Open Sans',
    pos=(0, -0.3), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-2.0);
accumulated_value = visual.TextBox2(
     win, text='', font='Open Sans',
     pos=(0, -0.3),     letterHeight=0.02,
     size=(0.25,0.1), borderWidth=4.0,
     color='black', colorSpace='rgb',
     opacity=None,
     bold=False, italic=False,
     lineSpacing=1.0,
     padding=0.0,
     anchor='center',
     fillColor=None, borderColor='black',
     flipHoriz=False, flipVert=False,
     editable=False,
     name='accumulated_value',
     autoLog=True,
)
text_9 = visual.TextStim(win=win, name='text_9',
    text='',
    font='Open Sans',
    pos=(0, 0.4), height=0.04, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-4.0);

# Initialize components for Routine "inter"
interClock = core.Clock()
image = visual.ImageStim(
    win=win,
    name='image', 
    image=None, mask=None,
    ori=0.0, pos=(0, 0), size=(0.5, 0.5),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=0.0)

# Initialize components for Routine "starttrial"
starttrialClock = core.Clock()
text_14 = visual.TextStim(win=win, name='text_14',
    text='¡Ya está preparado/a para empezar el experimento!\n\nHaga click en el ratón para continuar.',
    font='Open Sans',
    pos=(0.05, 0.05), height=0.05, wrapWidth=None, ori=0.2, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
mouse_8 = event.Mouse(win=win)
x, y = [None, None]
mouse_8.mouseClock = core.Clock()

# Initialize components for Routine "trial"
trialClock = core.Clock()
fix = visual.ShapeStim(
    win=win, name='fix', vertices='cross',
    size=[1.0, 1.0],
    ori=1.0, pos=[0,0],
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=0.0, interpolate=True)
etRecord_2 = hardware.eyetracker.EyetrackerControl(
    server=ioServer,
    tracker=eyetracker
)

# Initialize components for Routine "offer1"
offer1Clock = core.Clock()
rl1 = visual.Rect(
    win=win, name='rl1',
    width=(0.3, 0.5)[0], height=(0.3, 0.5)[1],
    ori=0.0, pos=(-0.4, 0),
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='dark grey',
    opacity=None, depth=0.0, interpolate=True)
cl1 = visual.Rect(
    win=win, name='cl1',
    width=[1.0, 1.0][0], height=[1.0, 1.0][1],
    ori=0.0, pos=[0,0],
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-1.0, interpolate=True)
etRecord_6 = hardware.eyetracker.EyetrackerControl(
    server=ioServer,
    tracker=eyetracker
)

# Initialize components for Routine "delay1"
delay1Clock = core.Clock()
image_2 = visual.ImageStim(
    win=win,
    name='image_2', 
    image=None, mask=None,
    ori=0.0, pos=(0, 0), size=(0.5, 0.5),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=0.0)
fix_2 = visual.ShapeStim(
    win=win, name='fix_2', vertices='cross',
    size=[1.0, 1.0],
    ori=1.0, pos=[0,0],
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-1.0, interpolate=True)

# Initialize components for Routine "offer2"
offer2Clock = core.Clock()
rr1 = visual.Rect(
    win=win, name='rr1',
    width=(0.3, 0.5)[0], height=(0.3, 0.5)[1],
    ori=0.0, pos=(0.4, 0),
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='dark grey',
    opacity=None, depth=0.0, interpolate=True)
cr1 = visual.Rect(
    win=win, name='cr1',
    width=[1.0, 1.0][0], height=[1.0, 1.0][1],
    ori=0.0, pos=[0,0],
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-1.0, interpolate=True)
etRecord_5 = hardware.eyetracker.EyetrackerControl(
    server=ioServer,
    tracker=eyetracker
)

# Initialize components for Routine "delay2"
delay2Clock = core.Clock()
image_3 = visual.ImageStim(
    win=win,
    name='image_3', 
    image=None, mask=None,
    ori=0.0, pos=(0, 0), size=(0.5, 0.5),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=0.0)
fix_3 = visual.ShapeStim(
    win=win, name='fix_3', vertices='cross',
    size=[1.0, 1.0],
    ori=1.0, pos=[0,0],
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-1.0, interpolate=True)

# Initialize components for Routine "observ"
observClock = core.Clock()
rl = visual.Rect(
    win=win, name='rl',units='height', 
    width=[1.0, 1.0][0], height=[1.0, 1.0][1],
    ori=0.0, pos=[0,0],
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='dark grey',
    opacity=None, depth=0.0, interpolate=True)
cl = visual.Rect(
    win=win, name='cl',units='height', 
    width=[1.0, 1.0][0], height=[1.0, 1.0][1],
    ori=0.0, pos=[0,0],
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-1.0, interpolate=True)
rr = visual.Rect(
    win=win, name='rr',units='height', 
    width=[1.0, 1.0][0], height=[1.0, 1.0][1],
    ori=0.0, pos=[0,0],
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='dark grey',
    opacity=None, depth=-2.0, interpolate=True)
cr = visual.Rect(
    win=win, name='cr',units='height', 
    width=[1.0, 1.0][0], height=[1.0, 1.0][1],
    ori=0.0, pos=[0,0],
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-3.0, interpolate=True)
etRecord_3 = hardware.eyetracker.EyetrackerControl(
    server=ioServer,
    tracker=eyetracker
)

# Initialize components for Routine "decision_trials"
decision_trialsClock = core.Clock()
decr_2 = visual.Rect(
    win=win, name='decr_2',
    width=[1.0, 1.0][0], height=[1.0, 1.0][1],
    ori=1.0, pos=[0,0],
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor=None,
    opacity=None, depth=0.0, interpolate=True)
dec2_2 = visual.Rect(
    win=win, name='dec2_2',
    width=[1.0, 1.0][0], height=[1.0, 1.0][1],
    ori=1.0, pos=[0,0],
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor=None,
    opacity=None, depth=-1.0, interpolate=True)
etRecord_7 = hardware.eyetracker.EyetrackerControl(
    server=ioServer,
    tracker=eyetracker
)
sound_2 = sound.Sound('A', secs=0.5, stereo=True, hamming=True,
    name='sound_2')
sound_2.setVolume(1.0)
mouse_7 = event.Mouse(win=win)
x, y = [None, None]
mouse_7.mouseClock = core.Clock()

# Initialize components for Routine "confidence"
confidenceClock = core.Clock()
slider = visual.Slider(win=win, name='slider',
    startValue=None, size=(1.0, 0.1), pos=(0, 0), units=None,
    labels=("-","+"), ticks=(1, 2, 3, 4, 5 , 6 ,7 ,8 , 9 ,10), granularity=1.0,
    style='radio', styleTweaks=(), opacity=None,
    color='white', fillColor='Red', borderColor='white', colorSpace='rgb',
    font='Open Sans', labelHeight=0.05,
    flip=False, depth=0, readOnly=False)
text_2 = visual.TextStim(win=win, name='text_2',
    text='¿Qué nivel de seguridad siente sobre la decisión que acaba de tomar?',
    font='Open Sans',
    pos=(0, 0.1), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-1.0);

# Initialize components for Routine "score"
scoreClock = core.Clock()
score1 = visual.TextStim(win=win, name='score1',
    text='',
    font='Open Sans',
    pos=(-0.1, 0.05), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
Score = visual.TextBox2(
     win, text='', font='Open Sans',
     pos=(0, 0.05),     letterHeight=0.02,
     size=(0.25,0.1), borderWidth=4.0,
     color='black', colorSpace='rgb',
     opacity=None,
     bold=False, italic=False,
     lineSpacing=1.0,
     padding=0.0,
     anchor='center',
     fillColor=None, borderColor='black',
     flipHoriz=False, flipVert=False,
     editable=False,
     name='Score',
     autoLog=True,
)
accumulatedvalue = visual.TextStim(win=win, name='accumulatedvalue',
    text='',
    font='Open Sans',
    pos=(0, -0.3), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-2.0);
accumulated_value = visual.TextBox2(
     win, text='', font='Open Sans',
     pos=(0, -0.3),     letterHeight=0.02,
     size=(0.25,0.1), borderWidth=4.0,
     color='black', colorSpace='rgb',
     opacity=None,
     bold=False, italic=False,
     lineSpacing=1.0,
     padding=0.0,
     anchor='center',
     fillColor=None, borderColor='black',
     flipHoriz=False, flipVert=False,
     editable=False,
     name='accumulated_value',
     autoLog=True,
)
text_9 = visual.TextStim(win=win, name='text_9',
    text='',
    font='Open Sans',
    pos=(0, 0.4), height=0.04, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-4.0);

# Initialize components for Routine "inter"
interClock = core.Clock()
image = visual.ImageStim(
    win=win,
    name='image', 
    image=None, mask=None,
    ori=0.0, pos=(0, 0), size=(0.5, 0.5),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=0.0)

# Initialize components for Routine "calfinal"
calfinalClock = core.Clock()
text_5 = visual.TextStim(win=win, name='text_5',
    text='',
    font='Open Sans',
    pos=(0, 0.1), height=0.1, wrapWidth=None, ori=0.0, 
    color='black', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
text_8 = visual.TextStim(win=win, name='text_8',
    text='',
    font='Open Sans',
    pos=(0, 0.35), height=0.1, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-1.0);

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine 

# ------Prepare to start Routine "instruction_for_caliberation_"-------
continueRoutine = True
# update component parameters for each repeat
# setup some python lists for storing info about the mouse_4
gotValidClick = False  # until a click is received
win.mouseVisible = False
# keep track of which components have finished
instruction_for_caliberation_Components = [text, mouse_4]
for thisComponent in instruction_for_caliberation_Components:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
instruction_for_caliberation_Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "instruction_for_caliberation_"-------
while continueRoutine:
    # get current time
    t = instruction_for_caliberation_Clock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=instruction_for_caliberation_Clock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *text* updates
    if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        text.frameNStart = frameN  # exact frame index
        text.tStart = t  # local t and not account for scr refresh
        text.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
        text.setAutoDraw(True)
    # *mouse_4* updates
    if mouse_4.status == NOT_STARTED and t >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        mouse_4.frameNStart = frameN  # exact frame index
        mouse_4.tStart = t  # local t and not account for scr refresh
        mouse_4.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(mouse_4, 'tStartRefresh')  # time at next scr refresh
        mouse_4.status = STARTED
        mouse_4.mouseClock.reset()
        prevButtonState = mouse_4.getPressed()  # if button is down already this ISN'T a new click
    if mouse_4.status == STARTED:  # only update if started and not finished!
        buttons = mouse_4.getPressed()
        if buttons != prevButtonState:  # button state changed?
            prevButtonState = buttons
            if sum(buttons) > 0:  # state changed to a new click
                # abort routine on response
                continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in instruction_for_caliberation_Components:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "instruction_for_caliberation_"-------
for thisComponent in instruction_for_caliberation_Components:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('text.started', text.tStartRefresh)
thisExp.addData('text.stopped', text.tStopRefresh)
# store data for thisExp (ExperimentHandler)
x, y = mouse_4.getPos()
buttons = mouse_4.getPressed()
thisExp.addData('mouse_4.x', x)
thisExp.addData('mouse_4.y', y)
thisExp.addData('mouse_4.leftButton', buttons[0])
thisExp.addData('mouse_4.midButton', buttons[1])
thisExp.addData('mouse_4.rightButton', buttons[2])
thisExp.addData('mouse_4.started', mouse_4.tStart)
thisExp.addData('mouse_4.stopped', mouse_4.tStop)
thisExp.nextEntry()
# the Routine "instruction_for_caliberation_" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "mouse_hidder"-------
continueRoutine = True
# update component parameters for each repeat
win.mouseVisible = False
# keep track of which components have finished
mouse_hidderComponents = []
for thisComponent in mouse_hidderComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
mouse_hidderClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "mouse_hidder"-------
while continueRoutine:
    # get current time
    t = mouse_hidderClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=mouse_hidderClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in mouse_hidderComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "mouse_hidder"-------
for thisComponent in mouse_hidderComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
win.mouseVisible = False
# the Routine "mouse_hidder" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# -------Run Routine 'calibration'-------

# define target for calibration
calibrationTarget = visual.TargetStim(win, 
    name='calibrationTarget',
    radius=0.01, fillColor='', borderColor='white', lineWidth=2.0,
    innerRadius=0.0035, innerFillColor='green', innerBorderColor='white', innerLineWidth=2.0,
    colorSpace='rgb', units=None
)
# define parameters for calibration
calibration = hardware.eyetracker.EyetrackerCalibration(win, 
    eyetracker, calibrationTarget,
    units=None, colorSpace='rgb',
    progressMode='time', targetDur=1.5, expandScale=1.5,
    targetLayout='THREE_POINTS', randomisePos=True,
    movementAnimation=True, targetDelay=1.0
)
# run calibration
calibration.run()
# clear any keypresses from during calibration so they don't interfere with the experiment
defaultKeyboard.clearEvents()
# the Routine "calibration" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# -------Run Routine 'validation'-------

# define target for validation
validationTarget = visual.TargetStim(win, 
    name='validationTarget',
    radius=0.01, fillColor='', borderColor='white', lineWidth=2.0,
    innerRadius=0.0035, innerFillColor='green', innerBorderColor='white', innerLineWidth=2.0,
    colorSpace='rgb', units=None
)
# define parameters for validation
validation = iohub.ValidationProcedure(win,
    target=validationTarget,
    gaze_cursor='green', 
    positions='THREE_POINTS', randomize_positions=True,
    expand_scale=1.5, target_duration=1.5,
    enable_position_animation=True, target_delay=1.0,
    progress_on_key=None,
    show_results_screen=True, save_results_screen=False,
    color_space='rgb', unit_type=None
)
# run validation
validation.run()
# clear any keypresses from during validation so they don't interfere with the experiment
defaultKeyboard.clearEvents()
# the Routine "validation" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "instruction_"-------
continueRoutine = True
# update component parameters for each repeat
# setup some python lists for storing info about the mouse_2
gotValidClick = False  # until a click is received
# keep track of which components have finished
instruction_Components = [text_3, mouse_2]
for thisComponent in instruction_Components:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
instruction_Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "instruction_"-------
while continueRoutine:
    # get current time
    t = instruction_Clock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=instruction_Clock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *text_3* updates
    if text_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        text_3.frameNStart = frameN  # exact frame index
        text_3.tStart = t  # local t and not account for scr refresh
        text_3.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text_3, 'tStartRefresh')  # time at next scr refresh
        text_3.setAutoDraw(True)
    # *mouse_2* updates
    if mouse_2.status == NOT_STARTED and t >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        mouse_2.frameNStart = frameN  # exact frame index
        mouse_2.tStart = t  # local t and not account for scr refresh
        mouse_2.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(mouse_2, 'tStartRefresh')  # time at next scr refresh
        mouse_2.status = STARTED
        mouse_2.mouseClock.reset()
        prevButtonState = mouse_2.getPressed()  # if button is down already this ISN'T a new click
    if mouse_2.status == STARTED:  # only update if started and not finished!
        buttons = mouse_2.getPressed()
        if buttons != prevButtonState:  # button state changed?
            prevButtonState = buttons
            if sum(buttons) > 0:  # state changed to a new click
                # abort routine on response
                continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in instruction_Components:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "instruction_"-------
for thisComponent in instruction_Components:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('text_3.started', text_3.tStartRefresh)
thisExp.addData('text_3.stopped', text_3.tStopRefresh)
# store data for thisExp (ExperimentHandler)
x, y = mouse_2.getPos()
buttons = mouse_2.getPressed()
thisExp.addData('mouse_2.x', x)
thisExp.addData('mouse_2.y', y)
thisExp.addData('mouse_2.leftButton', buttons[0])
thisExp.addData('mouse_2.midButton', buttons[1])
thisExp.addData('mouse_2.rightButton', buttons[2])
thisExp.addData('mouse_2.started', mouse_2.tStart)
thisExp.addData('mouse_2.stopped', mouse_2.tStop)
thisExp.nextEntry()
# the Routine "instruction_" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "instructions21"-------
continueRoutine = True
# update component parameters for each repeat
# setup some python lists for storing info about the mouse_5
gotValidClick = False  # until a click is received
examplecolor.setFillColor('green')
examplecolor.setPos((0, (-(0.35-0.175)/2)+0.15))
examplecolor.setSize((0.15, 0.175))
examplecolor.setLineColor('white')
# keep track of which components have finished
instructions21Components = [text_11, mouse_5, examplecolumn, examplecolor]
for thisComponent in instructions21Components:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
instructions21Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "instructions21"-------
while continueRoutine:
    # get current time
    t = instructions21Clock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=instructions21Clock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *text_11* updates
    if text_11.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        text_11.frameNStart = frameN  # exact frame index
        text_11.tStart = t  # local t and not account for scr refresh
        text_11.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text_11, 'tStartRefresh')  # time at next scr refresh
        text_11.setAutoDraw(True)
    # *mouse_5* updates
    if mouse_5.status == NOT_STARTED and t >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        mouse_5.frameNStart = frameN  # exact frame index
        mouse_5.tStart = t  # local t and not account for scr refresh
        mouse_5.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(mouse_5, 'tStartRefresh')  # time at next scr refresh
        mouse_5.status = STARTED
        mouse_5.mouseClock.reset()
        prevButtonState = mouse_5.getPressed()  # if button is down already this ISN'T a new click
    if mouse_5.status == STARTED:  # only update if started and not finished!
        buttons = mouse_5.getPressed()
        if buttons != prevButtonState:  # button state changed?
            prevButtonState = buttons
            if sum(buttons) > 0:  # state changed to a new click
                # abort routine on response
                continueRoutine = False
    
    # *examplecolumn* updates
    if examplecolumn.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        examplecolumn.frameNStart = frameN  # exact frame index
        examplecolumn.tStart = t  # local t and not account for scr refresh
        examplecolumn.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(examplecolumn, 'tStartRefresh')  # time at next scr refresh
        examplecolumn.setAutoDraw(True)
    
    # *examplecolor* updates
    if examplecolor.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        examplecolor.frameNStart = frameN  # exact frame index
        examplecolor.tStart = t  # local t and not account for scr refresh
        examplecolor.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(examplecolor, 'tStartRefresh')  # time at next scr refresh
        examplecolor.setAutoDraw(True)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in instructions21Components:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "instructions21"-------
for thisComponent in instructions21Components:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('text_11.started', text_11.tStartRefresh)
thisExp.addData('text_11.stopped', text_11.tStopRefresh)
# store data for thisExp (ExperimentHandler)
x, y = mouse_5.getPos()
buttons = mouse_5.getPressed()
thisExp.addData('mouse_5.x', x)
thisExp.addData('mouse_5.y', y)
thisExp.addData('mouse_5.leftButton', buttons[0])
thisExp.addData('mouse_5.midButton', buttons[1])
thisExp.addData('mouse_5.rightButton', buttons[2])
thisExp.addData('mouse_5.started', mouse_5.tStart)
thisExp.addData('mouse_5.stopped', mouse_5.tStop)
thisExp.nextEntry()
thisExp.addData('examplecolumn.started', examplecolumn.tStartRefresh)
thisExp.addData('examplecolumn.stopped', examplecolumn.tStopRefresh)
thisExp.addData('examplecolor.started', examplecolor.tStartRefresh)
thisExp.addData('examplecolor.stopped', examplecolor.tStopRefresh)
# the Routine "instructions21" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "instruction_2"-------
continueRoutine = True
# update component parameters for each repeat
# setup some python lists for storing info about the mouse_3
gotValidClick = False  # until a click is received
# keep track of which components have finished
instruction_2Components = [text_6, mouse_3]
for thisComponent in instruction_2Components:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
instruction_2Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "instruction_2"-------
while continueRoutine:
    # get current time
    t = instruction_2Clock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=instruction_2Clock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *text_6* updates
    if text_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        text_6.frameNStart = frameN  # exact frame index
        text_6.tStart = t  # local t and not account for scr refresh
        text_6.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text_6, 'tStartRefresh')  # time at next scr refresh
        text_6.setAutoDraw(True)
    # *mouse_3* updates
    if mouse_3.status == NOT_STARTED and t >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        mouse_3.frameNStart = frameN  # exact frame index
        mouse_3.tStart = t  # local t and not account for scr refresh
        mouse_3.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(mouse_3, 'tStartRefresh')  # time at next scr refresh
        mouse_3.status = STARTED
        mouse_3.mouseClock.reset()
        prevButtonState = mouse_3.getPressed()  # if button is down already this ISN'T a new click
    if mouse_3.status == STARTED:  # only update if started and not finished!
        buttons = mouse_3.getPressed()
        if buttons != prevButtonState:  # button state changed?
            prevButtonState = buttons
            if sum(buttons) > 0:  # state changed to a new click
                # abort routine on response
                continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in instruction_2Components:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "instruction_2"-------
for thisComponent in instruction_2Components:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('text_6.started', text_6.tStartRefresh)
thisExp.addData('text_6.stopped', text_6.tStopRefresh)
# store data for thisExp (ExperimentHandler)
x, y = mouse_3.getPos()
buttons = mouse_3.getPressed()
thisExp.addData('mouse_3.x', x)
thisExp.addData('mouse_3.y', y)
thisExp.addData('mouse_3.leftButton', buttons[0])
thisExp.addData('mouse_3.midButton', buttons[1])
thisExp.addData('mouse_3.rightButton', buttons[2])
thisExp.addData('mouse_3.started', mouse_3.tStart)
thisExp.addData('mouse_3.stopped', mouse_3.tStop)
thisExp.nextEntry()
# the Routine "instruction_2" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "instruction3"-------
continueRoutine = True
# update component parameters for each repeat
# setup some python lists for storing info about the mouse_6
gotValidClick = False  # until a click is received
# keep track of which components have finished
instruction3Components = [text_12, mouse_6]
for thisComponent in instruction3Components:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
instruction3Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "instruction3"-------
while continueRoutine:
    # get current time
    t = instruction3Clock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=instruction3Clock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *text_12* updates
    if text_12.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        text_12.frameNStart = frameN  # exact frame index
        text_12.tStart = t  # local t and not account for scr refresh
        text_12.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text_12, 'tStartRefresh')  # time at next scr refresh
        text_12.setAutoDraw(True)
    # *mouse_6* updates
    if mouse_6.status == NOT_STARTED and t >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        mouse_6.frameNStart = frameN  # exact frame index
        mouse_6.tStart = t  # local t and not account for scr refresh
        mouse_6.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(mouse_6, 'tStartRefresh')  # time at next scr refresh
        mouse_6.status = STARTED
        mouse_6.mouseClock.reset()
        prevButtonState = mouse_6.getPressed()  # if button is down already this ISN'T a new click
    if mouse_6.status == STARTED:  # only update if started and not finished!
        buttons = mouse_6.getPressed()
        if buttons != prevButtonState:  # button state changed?
            prevButtonState = buttons
            if sum(buttons) > 0:  # state changed to a new click
                # abort routine on response
                continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in instruction3Components:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "instruction3"-------
for thisComponent in instruction3Components:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('text_12.started', text_12.tStartRefresh)
thisExp.addData('text_12.stopped', text_12.tStopRefresh)
# store data for thisExp (ExperimentHandler)
x, y = mouse_6.getPos()
buttons = mouse_6.getPressed()
thisExp.addData('mouse_6.x', x)
thisExp.addData('mouse_6.y', y)
thisExp.addData('mouse_6.leftButton', buttons[0])
thisExp.addData('mouse_6.midButton', buttons[1])
thisExp.addData('mouse_6.rightButton', buttons[2])
thisExp.addData('mouse_6.started', mouse_6.tStart)
thisExp.addData('mouse_6.stopped', mouse_6.tStop)
thisExp.nextEntry()
# the Routine "instruction3" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "instructions4"-------
continueRoutine = True
# update component parameters for each repeat
# setup some python lists for storing info about the mouse_10
gotValidClick = False  # until a click is received
fix_instructions.setPos((0, 0))
fix_instructions.setSize((0.05, 0.05))
fix_instructions.setOri(0.0)
# keep track of which components have finished
instructions4Components = [text_17, mouse_10, fix_instructions]
for thisComponent in instructions4Components:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
instructions4Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "instructions4"-------
while continueRoutine:
    # get current time
    t = instructions4Clock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=instructions4Clock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *text_17* updates
    if text_17.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        text_17.frameNStart = frameN  # exact frame index
        text_17.tStart = t  # local t and not account for scr refresh
        text_17.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text_17, 'tStartRefresh')  # time at next scr refresh
        text_17.setAutoDraw(True)
    # *mouse_10* updates
    if mouse_10.status == NOT_STARTED and t >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        mouse_10.frameNStart = frameN  # exact frame index
        mouse_10.tStart = t  # local t and not account for scr refresh
        mouse_10.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(mouse_10, 'tStartRefresh')  # time at next scr refresh
        mouse_10.status = STARTED
        mouse_10.mouseClock.reset()
        prevButtonState = mouse_10.getPressed()  # if button is down already this ISN'T a new click
    if mouse_10.status == STARTED:  # only update if started and not finished!
        buttons = mouse_10.getPressed()
        if buttons != prevButtonState:  # button state changed?
            prevButtonState = buttons
            if sum(buttons) > 0:  # state changed to a new click
                # abort routine on response
                continueRoutine = False
    
    # *fix_instructions* updates
    if fix_instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        fix_instructions.frameNStart = frameN  # exact frame index
        fix_instructions.tStart = t  # local t and not account for scr refresh
        fix_instructions.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(fix_instructions, 'tStartRefresh')  # time at next scr refresh
        fix_instructions.setAutoDraw(True)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in instructions4Components:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "instructions4"-------
for thisComponent in instructions4Components:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('text_17.started', text_17.tStartRefresh)
thisExp.addData('text_17.stopped', text_17.tStopRefresh)
# store data for thisExp (ExperimentHandler)
x, y = mouse_10.getPos()
buttons = mouse_10.getPressed()
thisExp.addData('mouse_10.x', x)
thisExp.addData('mouse_10.y', y)
thisExp.addData('mouse_10.leftButton', buttons[0])
thisExp.addData('mouse_10.midButton', buttons[1])
thisExp.addData('mouse_10.rightButton', buttons[2])
thisExp.addData('mouse_10.started', mouse_10.tStart)
thisExp.addData('mouse_10.stopped', mouse_10.tStop)
thisExp.nextEntry()
thisExp.addData('fix_instructions.started', fix_instructions.tStartRefresh)
thisExp.addData('fix_instructions.stopped', fix_instructions.tStopRefresh)
# the Routine "instructions4" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "starttraining"-------
continueRoutine = True
# update component parameters for each repeat
# setup some python lists for storing info about the mouse_9
gotValidClick = False  # until a click is received
# keep track of which components have finished
starttrainingComponents = [text_16, mouse_9]
for thisComponent in starttrainingComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
starttrainingClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "starttraining"-------
while continueRoutine:
    # get current time
    t = starttrainingClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=starttrainingClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *text_16* updates
    if text_16.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        text_16.frameNStart = frameN  # exact frame index
        text_16.tStart = t  # local t and not account for scr refresh
        text_16.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text_16, 'tStartRefresh')  # time at next scr refresh
        text_16.setAutoDraw(True)
    # *mouse_9* updates
    if mouse_9.status == NOT_STARTED and t >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        mouse_9.frameNStart = frameN  # exact frame index
        mouse_9.tStart = t  # local t and not account for scr refresh
        mouse_9.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(mouse_9, 'tStartRefresh')  # time at next scr refresh
        mouse_9.status = STARTED
        mouse_9.mouseClock.reset()
        prevButtonState = mouse_9.getPressed()  # if button is down already this ISN'T a new click
    if mouse_9.status == STARTED:  # only update if started and not finished!
        buttons = mouse_9.getPressed()
        if buttons != prevButtonState:  # button state changed?
            prevButtonState = buttons
            if sum(buttons) > 0:  # state changed to a new click
                # abort routine on response
                continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in starttrainingComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "starttraining"-------
for thisComponent in starttrainingComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('text_16.started', text_16.tStartRefresh)
thisExp.addData('text_16.stopped', text_16.tStopRefresh)
# store data for thisExp (ExperimentHandler)
x, y = mouse_9.getPos()
buttons = mouse_9.getPressed()
thisExp.addData('mouse_9.x', x)
thisExp.addData('mouse_9.y', y)
thisExp.addData('mouse_9.leftButton', buttons[0])
thisExp.addData('mouse_9.midButton', buttons[1])
thisExp.addData('mouse_9.rightButton', buttons[2])
thisExp.addData('mouse_9.started', mouse_9.tStart)
thisExp.addData('mouse_9.stopped', mouse_9.tStop)
thisExp.nextEntry()
# the Routine "starttraining" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "calc_2"-------
continueRoutine = True
# update component parameters for each repeat
# keep track of which components have finished
calc_2Components = []
for thisComponent in calc_2Components:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
calc_2Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "calc_2"-------
while continueRoutine:
    # get current time
    t = calc_2Clock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=calc_2Clock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in calc_2Components:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "calc_2"-------
for thisComponent in calc_2Components:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# the Routine "calc_2" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of conditions etc
training = data.TrialHandler(nReps=1.0, method='random', 
    extraInfo=expInfo, originPath=-1,
    trialList=data.importConditions('training.xlsx', selection='[0:1]'),
    seed=None, name='training')
thisExp.addLoop(training)  # add the loop to the experiment
thisTraining = training.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisTraining.rgb)
if thisTraining != None:
    for paramName in thisTraining:
        exec('{} = thisTraining[paramName]'.format(paramName))

for thisTraining in training:
    currentLoop = training
    # abbreviate parameter names if possible (e.g. rgb = thisTraining.rgb)
    if thisTraining != None:
        for paramName in thisTraining:
            exec('{} = thisTraining[paramName]'.format(paramName))
    
    # ------Prepare to start Routine "trial"-------
    continueRoutine = True
    routineTimer.add(1.000000)
    # update component parameters for each repeat
    fix.setPos((0, 0))
    fix.setSize((0.05, 0.05))
    fix.setOri(0.0)
    #arduino_board.digital[arduino_pin].write(1)
    #arduino_board.pass_time(0.1)
    #arduino_board.digital[arduino_pin].write(0)
    # keep track of which components have finished
    trialComponents = [fix, etRecord_2]
    for thisComponent in trialComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    trialClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "trial"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = trialClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=trialClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *fix* updates
        if fix.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            fix.frameNStart = frameN  # exact frame index
            fix.tStart = t  # local t and not account for scr refresh
            fix.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fix, 'tStartRefresh')  # time at next scr refresh
            fix.setAutoDraw(True)
        if fix.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > fix.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                fix.tStop = t  # not accounting for scr refresh
                fix.frameNStop = frameN  # exact frame index
                win.timeOnFlip(fix, 'tStopRefresh')  # time at next scr refresh
                fix.setAutoDraw(False)
        # *etRecord_2* updates
        if etRecord_2.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            etRecord_2.frameNStart = frameN  # exact frame index
            etRecord_2.tStart = t  # local t and not account for scr refresh
            etRecord_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(etRecord_2, 'tStartRefresh')  # time at next scr refresh
            etRecord_2.status = STARTED
        if etRecord_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > etRecord_2.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                etRecord_2.tStop = t  # not accounting for scr refresh
                etRecord_2.frameNStop = frameN  # exact frame index
                win.timeOnFlip(etRecord_2, 'tStopRefresh')  # time at next scr refresh
                etRecord_2.status = FINISHED
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in trialComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "trial"-------
    for thisComponent in trialComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    training.addData('fix.started', fix.tStartRefresh)
    training.addData('fix.stopped', fix.tStopRefresh)
    # make sure the eyetracker recording stops
    if etRecord_2.status != FINISHED:
        etRecord_2.status = FINISHED
    
    # ------Prepare to start Routine "offer1_training"-------
    continueRoutine = True
    routineTimer.add(2.000000)
    # update component parameters for each repeat
    cl1_2.setFillColor(colorl)
    cl1_2.setPos((-0.4, -(0.5-heightl)/2))
    cl1_2.setSize((0.3, heightl))
    cl1_2.setLineColor('white')
    text_13.setPos((-0.4, -(0.5-heightl)/2))
    text_13.setText(color_value[colorl])
    # keep track of which components have finished
    offer1_trainingComponents = [rl1_2, cl1_2, etRecord_8, text_13]
    for thisComponent in offer1_trainingComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    offer1_trainingClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "offer1_training"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = offer1_trainingClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=offer1_trainingClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *rl1_2* updates
        if rl1_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            rl1_2.frameNStart = frameN  # exact frame index
            rl1_2.tStart = t  # local t and not account for scr refresh
            rl1_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(rl1_2, 'tStartRefresh')  # time at next scr refresh
            rl1_2.setAutoDraw(True)
        if rl1_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > rl1_2.tStartRefresh + 2-frameTolerance:
                # keep track of stop time/frame for later
                rl1_2.tStop = t  # not accounting for scr refresh
                rl1_2.frameNStop = frameN  # exact frame index
                win.timeOnFlip(rl1_2, 'tStopRefresh')  # time at next scr refresh
                rl1_2.setAutoDraw(False)
        
        # *cl1_2* updates
        if cl1_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            cl1_2.frameNStart = frameN  # exact frame index
            cl1_2.tStart = t  # local t and not account for scr refresh
            cl1_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(cl1_2, 'tStartRefresh')  # time at next scr refresh
            cl1_2.setAutoDraw(True)
        if cl1_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > cl1_2.tStartRefresh + 2-frameTolerance:
                # keep track of stop time/frame for later
                cl1_2.tStop = t  # not accounting for scr refresh
                cl1_2.frameNStop = frameN  # exact frame index
                win.timeOnFlip(cl1_2, 'tStopRefresh')  # time at next scr refresh
                cl1_2.setAutoDraw(False)
        # *etRecord_8* updates
        if etRecord_8.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            etRecord_8.frameNStart = frameN  # exact frame index
            etRecord_8.tStart = t  # local t and not account for scr refresh
            etRecord_8.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(etRecord_8, 'tStartRefresh')  # time at next scr refresh
            etRecord_8.status = STARTED
        if etRecord_8.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > etRecord_8.tStartRefresh + 2-frameTolerance:
                # keep track of stop time/frame for later
                etRecord_8.tStop = t  # not accounting for scr refresh
                etRecord_8.frameNStop = frameN  # exact frame index
                win.timeOnFlip(etRecord_8, 'tStopRefresh')  # time at next scr refresh
                etRecord_8.status = FINISHED
        
        # *text_13* updates
        if text_13.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_13.frameNStart = frameN  # exact frame index
            text_13.tStart = t  # local t and not account for scr refresh
            text_13.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_13, 'tStartRefresh')  # time at next scr refresh
            text_13.setAutoDraw(True)
        if text_13.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_13.tStartRefresh + 2.0-frameTolerance:
                # keep track of stop time/frame for later
                text_13.tStop = t  # not accounting for scr refresh
                text_13.frameNStop = frameN  # exact frame index
                win.timeOnFlip(text_13, 'tStopRefresh')  # time at next scr refresh
                text_13.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in offer1_trainingComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "offer1_training"-------
    for thisComponent in offer1_trainingComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    training.addData('rl1_2.started', rl1_2.tStartRefresh)
    training.addData('rl1_2.stopped', rl1_2.tStopRefresh)
    training.addData('cl1_2.started', cl1_2.tStartRefresh)
    training.addData('cl1_2.stopped', cl1_2.tStopRefresh)
    # make sure the eyetracker recording stops
    if etRecord_8.status != FINISHED:
        etRecord_8.status = FINISHED
    training.addData('text_13.started', text_13.tStartRefresh)
    training.addData('text_13.stopped', text_13.tStopRefresh)
    
    # ------Prepare to start Routine "delay1"-------
    continueRoutine = True
    routineTimer.add(2.000000)
    # update component parameters for each repeat
    fix_2.setPos((0, 0))
    fix_2.setSize((0.05, 0.05))
    fix_2.setOri(0.0)
    # keep track of which components have finished
    delay1Components = [image_2, fix_2]
    for thisComponent in delay1Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    delay1Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "delay1"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = delay1Clock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=delay1Clock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *image_2* updates
        if image_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            image_2.frameNStart = frameN  # exact frame index
            image_2.tStart = t  # local t and not account for scr refresh
            image_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_2, 'tStartRefresh')  # time at next scr refresh
            image_2.setAutoDraw(True)
        if image_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > image_2.tStartRefresh + 2-frameTolerance:
                # keep track of stop time/frame for later
                image_2.tStop = t  # not accounting for scr refresh
                image_2.frameNStop = frameN  # exact frame index
                win.timeOnFlip(image_2, 'tStopRefresh')  # time at next scr refresh
                image_2.setAutoDraw(False)
        
        # *fix_2* updates
        if fix_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            fix_2.frameNStart = frameN  # exact frame index
            fix_2.tStart = t  # local t and not account for scr refresh
            fix_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fix_2, 'tStartRefresh')  # time at next scr refresh
            fix_2.setAutoDraw(True)
        if fix_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > fix_2.tStartRefresh + 2-frameTolerance:
                # keep track of stop time/frame for later
                fix_2.tStop = t  # not accounting for scr refresh
                fix_2.frameNStop = frameN  # exact frame index
                win.timeOnFlip(fix_2, 'tStopRefresh')  # time at next scr refresh
                fix_2.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in delay1Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "delay1"-------
    for thisComponent in delay1Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    training.addData('image_2.started', image_2.tStartRefresh)
    training.addData('image_2.stopped', image_2.tStopRefresh)
    training.addData('fix_2.started', fix_2.tStartRefresh)
    training.addData('fix_2.stopped', fix_2.tStopRefresh)
    
    # ------Prepare to start Routine "offer2_training"-------
    continueRoutine = True
    routineTimer.add(2.000000)
    # update component parameters for each repeat
    cr1_2.setFillColor(colorr)
    cr1_2.setPos((0.4 , -(0.5-heightr)/2))
    cr1_2.setSize((0.3, heightr))
    cr1_2.setLineColor('white')
    text14.setPos((0.4, -(0.5-heightr)/2))
    text14.setText(color_value[colorr])
    # keep track of which components have finished
    offer2_trainingComponents = [rr1_2, cr1_2, etRecord_9, text14]
    for thisComponent in offer2_trainingComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    offer2_trainingClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "offer2_training"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = offer2_trainingClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=offer2_trainingClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *rr1_2* updates
        if rr1_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            rr1_2.frameNStart = frameN  # exact frame index
            rr1_2.tStart = t  # local t and not account for scr refresh
            rr1_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(rr1_2, 'tStartRefresh')  # time at next scr refresh
            rr1_2.setAutoDraw(True)
        if rr1_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > rr1_2.tStartRefresh + 2-frameTolerance:
                # keep track of stop time/frame for later
                rr1_2.tStop = t  # not accounting for scr refresh
                rr1_2.frameNStop = frameN  # exact frame index
                win.timeOnFlip(rr1_2, 'tStopRefresh')  # time at next scr refresh
                rr1_2.setAutoDraw(False)
        
        # *cr1_2* updates
        if cr1_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            cr1_2.frameNStart = frameN  # exact frame index
            cr1_2.tStart = t  # local t and not account for scr refresh
            cr1_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(cr1_2, 'tStartRefresh')  # time at next scr refresh
            cr1_2.setAutoDraw(True)
        if cr1_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > cr1_2.tStartRefresh + 2-frameTolerance:
                # keep track of stop time/frame for later
                cr1_2.tStop = t  # not accounting for scr refresh
                cr1_2.frameNStop = frameN  # exact frame index
                win.timeOnFlip(cr1_2, 'tStopRefresh')  # time at next scr refresh
                cr1_2.setAutoDraw(False)
        # *etRecord_9* updates
        if etRecord_9.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            etRecord_9.frameNStart = frameN  # exact frame index
            etRecord_9.tStart = t  # local t and not account for scr refresh
            etRecord_9.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(etRecord_9, 'tStartRefresh')  # time at next scr refresh
            etRecord_9.status = STARTED
        if etRecord_9.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > etRecord_9.tStartRefresh + 2-frameTolerance:
                # keep track of stop time/frame for later
                etRecord_9.tStop = t  # not accounting for scr refresh
                etRecord_9.frameNStop = frameN  # exact frame index
                win.timeOnFlip(etRecord_9, 'tStopRefresh')  # time at next scr refresh
                etRecord_9.status = FINISHED
        
        # *text14* updates
        if text14.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text14.frameNStart = frameN  # exact frame index
            text14.tStart = t  # local t and not account for scr refresh
            text14.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text14, 'tStartRefresh')  # time at next scr refresh
            text14.setAutoDraw(True)
        if text14.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text14.tStartRefresh + 2.0-frameTolerance:
                # keep track of stop time/frame for later
                text14.tStop = t  # not accounting for scr refresh
                text14.frameNStop = frameN  # exact frame index
                win.timeOnFlip(text14, 'tStopRefresh')  # time at next scr refresh
                text14.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in offer2_trainingComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "offer2_training"-------
    for thisComponent in offer2_trainingComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    training.addData('rr1_2.started', rr1_2.tStartRefresh)
    training.addData('rr1_2.stopped', rr1_2.tStopRefresh)
    training.addData('cr1_2.started', cr1_2.tStartRefresh)
    training.addData('cr1_2.stopped', cr1_2.tStopRefresh)
    # make sure the eyetracker recording stops
    if etRecord_9.status != FINISHED:
        etRecord_9.status = FINISHED
    training.addData('text14.started', text14.tStartRefresh)
    training.addData('text14.stopped', text14.tStopRefresh)
    
    # ------Prepare to start Routine "delay2"-------
    continueRoutine = True
    routineTimer.add(2.000000)
    # update component parameters for each repeat
    fix_3.setPos((0, 0))
    fix_3.setSize((0.05, 0.05))
    fix_3.setOri(0.0)
    # keep track of which components have finished
    delay2Components = [image_3, fix_3]
    for thisComponent in delay2Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    delay2Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "delay2"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = delay2Clock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=delay2Clock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *image_3* updates
        if image_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            image_3.frameNStart = frameN  # exact frame index
            image_3.tStart = t  # local t and not account for scr refresh
            image_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_3, 'tStartRefresh')  # time at next scr refresh
            image_3.setAutoDraw(True)
        if image_3.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > image_3.tStartRefresh + 2-frameTolerance:
                # keep track of stop time/frame for later
                image_3.tStop = t  # not accounting for scr refresh
                image_3.frameNStop = frameN  # exact frame index
                win.timeOnFlip(image_3, 'tStopRefresh')  # time at next scr refresh
                image_3.setAutoDraw(False)
        
        # *fix_3* updates
        if fix_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            fix_3.frameNStart = frameN  # exact frame index
            fix_3.tStart = t  # local t and not account for scr refresh
            fix_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fix_3, 'tStartRefresh')  # time at next scr refresh
            fix_3.setAutoDraw(True)
        if fix_3.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > fix_3.tStartRefresh + 2-frameTolerance:
                # keep track of stop time/frame for later
                fix_3.tStop = t  # not accounting for scr refresh
                fix_3.frameNStop = frameN  # exact frame index
                win.timeOnFlip(fix_3, 'tStopRefresh')  # time at next scr refresh
                fix_3.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in delay2Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "delay2"-------
    for thisComponent in delay2Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    training.addData('image_3.started', image_3.tStartRefresh)
    training.addData('image_3.stopped', image_3.tStopRefresh)
    training.addData('fix_3.started', fix_3.tStartRefresh)
    training.addData('fix_3.stopped', fix_3.tStopRefresh)
    
    # ------Prepare to start Routine "observ_training"-------
    continueRoutine = True
    routineTimer.add(2.000000)
    # update component parameters for each repeat
    rl_2.setPos((-0.5, 0))
    rl_2.setSize((0.3, 0.5))
    cl_2.setFillColor(colorl)
    cl_2.setPos((-0.5, -(0.5-heightl)/2))
    cl_2.setSize((0.3, heightl))
    rr_2.setPos((0.5, 0))
    rr_2.setSize((0.3, 0.5))
    cr_2.setFillColor(colorr)
    cr_2.setPos((0.5 , -(0.5-heightr)/2))
    cr_2.setSize((0.3, heightr))
    text_15.setPos((-0.5, -(0.5-heightl)/2))
    text_15.setText(color_value[colorl])
    text14_3.setPos((0.5, -(0.5-heightr)/2))
    text14_3.setText(color_value[colorr])
    # keep track of which components have finished
    observ_trainingComponents = [rl_2, cl_2, rr_2, cr_2, etRecord_10, text_15, text14_3]
    for thisComponent in observ_trainingComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    observ_trainingClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "observ_training"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = observ_trainingClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=observ_trainingClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *rl_2* updates
        if rl_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            rl_2.frameNStart = frameN  # exact frame index
            rl_2.tStart = t  # local t and not account for scr refresh
            rl_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(rl_2, 'tStartRefresh')  # time at next scr refresh
            rl_2.setAutoDraw(True)
        if rl_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > rl_2.tStartRefresh + 2-frameTolerance:
                # keep track of stop time/frame for later
                rl_2.tStop = t  # not accounting for scr refresh
                rl_2.frameNStop = frameN  # exact frame index
                win.timeOnFlip(rl_2, 'tStopRefresh')  # time at next scr refresh
                rl_2.setAutoDraw(False)
        
        # *cl_2* updates
        if cl_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            cl_2.frameNStart = frameN  # exact frame index
            cl_2.tStart = t  # local t and not account for scr refresh
            cl_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(cl_2, 'tStartRefresh')  # time at next scr refresh
            cl_2.setAutoDraw(True)
        if cl_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > cl_2.tStartRefresh + 2-frameTolerance:
                # keep track of stop time/frame for later
                cl_2.tStop = t  # not accounting for scr refresh
                cl_2.frameNStop = frameN  # exact frame index
                win.timeOnFlip(cl_2, 'tStopRefresh')  # time at next scr refresh
                cl_2.setAutoDraw(False)
        
        # *rr_2* updates
        if rr_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            rr_2.frameNStart = frameN  # exact frame index
            rr_2.tStart = t  # local t and not account for scr refresh
            rr_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(rr_2, 'tStartRefresh')  # time at next scr refresh
            rr_2.setAutoDraw(True)
        if rr_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > rr_2.tStartRefresh + 2-frameTolerance:
                # keep track of stop time/frame for later
                rr_2.tStop = t  # not accounting for scr refresh
                rr_2.frameNStop = frameN  # exact frame index
                win.timeOnFlip(rr_2, 'tStopRefresh')  # time at next scr refresh
                rr_2.setAutoDraw(False)
        
        # *cr_2* updates
        if cr_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            cr_2.frameNStart = frameN  # exact frame index
            cr_2.tStart = t  # local t and not account for scr refresh
            cr_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(cr_2, 'tStartRefresh')  # time at next scr refresh
            cr_2.setAutoDraw(True)
        if cr_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > cr_2.tStartRefresh + 2-frameTolerance:
                # keep track of stop time/frame for later
                cr_2.tStop = t  # not accounting for scr refresh
                cr_2.frameNStop = frameN  # exact frame index
                win.timeOnFlip(cr_2, 'tStopRefresh')  # time at next scr refresh
                cr_2.setAutoDraw(False)
        # *etRecord_10* updates
        if etRecord_10.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            etRecord_10.frameNStart = frameN  # exact frame index
            etRecord_10.tStart = t  # local t and not account for scr refresh
            etRecord_10.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(etRecord_10, 'tStartRefresh')  # time at next scr refresh
            etRecord_10.status = STARTED
        if etRecord_10.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > etRecord_10.tStartRefresh + 2-frameTolerance:
                # keep track of stop time/frame for later
                etRecord_10.tStop = t  # not accounting for scr refresh
                etRecord_10.frameNStop = frameN  # exact frame index
                win.timeOnFlip(etRecord_10, 'tStopRefresh')  # time at next scr refresh
                etRecord_10.status = FINISHED
        
        # *text_15* updates
        if text_15.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_15.frameNStart = frameN  # exact frame index
            text_15.tStart = t  # local t and not account for scr refresh
            text_15.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_15, 'tStartRefresh')  # time at next scr refresh
            text_15.setAutoDraw(True)
        if text_15.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_15.tStartRefresh + 2.0-frameTolerance:
                # keep track of stop time/frame for later
                text_15.tStop = t  # not accounting for scr refresh
                text_15.frameNStop = frameN  # exact frame index
                win.timeOnFlip(text_15, 'tStopRefresh')  # time at next scr refresh
                text_15.setAutoDraw(False)
        
        # *text14_3* updates
        if text14_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text14_3.frameNStart = frameN  # exact frame index
            text14_3.tStart = t  # local t and not account for scr refresh
            text14_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text14_3, 'tStartRefresh')  # time at next scr refresh
            text14_3.setAutoDraw(True)
        if text14_3.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text14_3.tStartRefresh + 2.0-frameTolerance:
                # keep track of stop time/frame for later
                text14_3.tStop = t  # not accounting for scr refresh
                text14_3.frameNStop = frameN  # exact frame index
                win.timeOnFlip(text14_3, 'tStopRefresh')  # time at next scr refresh
                text14_3.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in observ_trainingComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "observ_training"-------
    for thisComponent in observ_trainingComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    training.addData('rl_2.started', rl_2.tStartRefresh)
    training.addData('rl_2.stopped', rl_2.tStopRefresh)
    training.addData('cl_2.started', cl_2.tStartRefresh)
    training.addData('cl_2.stopped', cl_2.tStopRefresh)
    training.addData('rr_2.started', rr_2.tStartRefresh)
    training.addData('rr_2.stopped', rr_2.tStopRefresh)
    training.addData('cr_2.started', cr_2.tStartRefresh)
    training.addData('cr_2.stopped', cr_2.tStopRefresh)
    # make sure the eyetracker recording stops
    if etRecord_10.status != FINISHED:
        etRecord_10.status = FINISHED
    training.addData('text_15.started', text_15.tStartRefresh)
    training.addData('text_15.stopped', text_15.tStopRefresh)
    training.addData('text14_3.started', text14_3.tStartRefresh)
    training.addData('text14_3.stopped', text14_3.tStopRefresh)
    
    # ------Prepare to start Routine "decision"-------
    continueRoutine = True
    routineTimer.add(30.000000)
    # update component parameters for each repeat
    decr.setFillColor('none')
    decr.setOpacity(None)
    decr.setContrast(1.0)
    decr.setPos((-0.5 ,0))
    decr.setSize((0.3, 0.5))
    decr.setOri(0.0)
    decr.setLineColor('white')
    decr.setLineWidth(1.0)
    dec2.setPos((0.5 ,0))
    dec2.setSize((0.3, 0.5))
    dec2.setOri(0.0)
    sound_1.setSound('A', secs=0.5, hamming=True)
    sound_1.setVolume(1.0, log=False)
    # setup some python lists for storing info about the mouse
    gotValidClick = False  # until a click is received
    # keep track of which components have finished
    decisionComponents = [decr, dec2, etRecord_4, sound_1, mouse]
    for thisComponent in decisionComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    decisionClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "decision"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = decisionClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=decisionClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *decr* updates
        if decr.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            decr.frameNStart = frameN  # exact frame index
            decr.tStart = t  # local t and not account for scr refresh
            decr.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(decr, 'tStartRefresh')  # time at next scr refresh
            decr.setAutoDraw(True)
        if decr.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > decr.tStartRefresh + 30-frameTolerance:
                # keep track of stop time/frame for later
                decr.tStop = t  # not accounting for scr refresh
                decr.frameNStop = frameN  # exact frame index
                win.timeOnFlip(decr, 'tStopRefresh')  # time at next scr refresh
                decr.setAutoDraw(False)
        
        # *dec2* updates
        if dec2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            dec2.frameNStart = frameN  # exact frame index
            dec2.tStart = t  # local t and not account for scr refresh
            dec2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(dec2, 'tStartRefresh')  # time at next scr refresh
            dec2.setAutoDraw(True)
        if dec2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > dec2.tStartRefresh + 30-frameTolerance:
                # keep track of stop time/frame for later
                dec2.tStop = t  # not accounting for scr refresh
                dec2.frameNStop = frameN  # exact frame index
                win.timeOnFlip(dec2, 'tStopRefresh')  # time at next scr refresh
                dec2.setAutoDraw(False)
        # *etRecord_4* updates
        if etRecord_4.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            etRecord_4.frameNStart = frameN  # exact frame index
            etRecord_4.tStart = t  # local t and not account for scr refresh
            etRecord_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(etRecord_4, 'tStartRefresh')  # time at next scr refresh
            etRecord_4.status = STARTED
        if etRecord_4.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > etRecord_4.tStartRefresh + 30-frameTolerance:
                # keep track of stop time/frame for later
                etRecord_4.tStop = t  # not accounting for scr refresh
                etRecord_4.frameNStop = frameN  # exact frame index
                win.timeOnFlip(etRecord_4, 'tStopRefresh')  # time at next scr refresh
                etRecord_4.status = FINISHED
        # start/stop sound_1
        if sound_1.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
            # keep track of start time/frame for later
            sound_1.frameNStart = frameN  # exact frame index
            sound_1.tStart = t  # local t and not account for scr refresh
            sound_1.tStartRefresh = tThisFlipGlobal  # on global time
            sound_1.play(when=win)  # sync with win flip
        if sound_1.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > sound_1.tStartRefresh + 0.5-frameTolerance:
                # keep track of stop time/frame for later
                sound_1.tStop = t  # not accounting for scr refresh
                sound_1.frameNStop = frameN  # exact frame index
                win.timeOnFlip(sound_1, 'tStopRefresh')  # time at next scr refresh
                sound_1.stop()
        # *mouse* updates
        if mouse.status == NOT_STARTED and t >= 2-frameTolerance:
            # keep track of start time/frame for later
            mouse.frameNStart = frameN  # exact frame index
            mouse.tStart = t  # local t and not account for scr refresh
            mouse.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse, 'tStartRefresh')  # time at next scr refresh
            mouse.status = STARTED
            mouse.mouseClock.reset()
            prevButtonState = mouse.getPressed()  # if button is down already this ISN'T a new click
        if mouse.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > mouse.tStartRefresh + 28-frameTolerance:
                # keep track of stop time/frame for later
                mouse.tStop = t  # not accounting for scr refresh
                mouse.frameNStop = frameN  # exact frame index
                win.timeOnFlip(mouse, 'tStopRefresh')  # time at next scr refresh
                mouse.status = FINISHED
        if mouse.status == STARTED:  # only update if started and not finished!
            buttons = mouse.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # abort routine on response
                    continueRoutine = False
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in decisionComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "decision"-------
    for thisComponent in decisionComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    training.addData('decr.started', decr.tStartRefresh)
    training.addData('decr.stopped', decr.tStopRefresh)
    training.addData('dec2.started', dec2.tStartRefresh)
    training.addData('dec2.stopped', dec2.tStopRefresh)
    # make sure the eyetracker recording stops
    if etRecord_4.status != FINISHED:
        etRecord_4.status = FINISHED
    color_value={"lightgrey":5, "blue":7,"green":10}
    Er = heightr*2* color_value[colorr];
    El = heightl*2* color_value[colorl]
    import math as mt
    # mouse.rightButton ==1 (means right) .leftButton==1(means left)
    if (mouse.getPressed()[2] ==1 and Er > El) or (mouse.getPressed()[0]==1  and El > Er): 
        true_choice= true_choice+1
    
    
    mouse_l = 0
    mouse_r= 0
    
    if mouse.getPressed()[0] ==1:
        trial_value= np.random.choice([0,1],p=[(1-heightl / 0.5),(heightl / 0.5)])*color_value[colorl]
        text_to_show=trial_value
        mouse_l=1
        value_trial = trial_value+value_trial 
        print("mouse 1")
    elif mouse.getPressed()[2] ==1 :
        trial_value= np.random.choice([0,1],p=[(1-heightr / 0.5),(heightr / 0.5)])*color_value[colorr]
        text_to_show=trial_value
        value_trial  = trial_value+value_trial
        mouse_r=1
        print("mouse 2")
    
    percentage='Ha realizado el '+str(np.round((training.thisN+1)*100/training.nTotal))+'% del entrenamiento!'
    
    mouse.setPos([0,1])
    sound_1.stop()  # ensure sound has stopped at end of routine
    training.addData('sound_1.started', sound_1.tStartRefresh)
    training.addData('sound_1.stopped', sound_1.tStopRefresh)
    # store data for training (TrialHandler)
    x, y = mouse.getPos()
    buttons = mouse.getPressed()
    training.addData('mouse.x', x)
    training.addData('mouse.y', y)
    training.addData('mouse.leftButton', buttons[0])
    training.addData('mouse.midButton', buttons[1])
    training.addData('mouse.rightButton', buttons[2])
    training.addData('mouse.started', mouse.tStart)
    training.addData('mouse.stopped', mouse.tStop)
    
    # ------Prepare to start Routine "confidence"-------
    continueRoutine = True
    # update component parameters for each repeat
    slider.reset()
    win.mouseVisible = True
    mouse.setPos([0,1])
    # keep track of which components have finished
    confidenceComponents = [slider, text_2]
    for thisComponent in confidenceComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    confidenceClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "confidence"-------
    while continueRoutine:
        # get current time
        t = confidenceClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=confidenceClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *slider* updates
        if slider.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            slider.frameNStart = frameN  # exact frame index
            slider.tStart = t  # local t and not account for scr refresh
            slider.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(slider, 'tStartRefresh')  # time at next scr refresh
            slider.setAutoDraw(True)
        
        # Check slider for response to end routine
        if slider.getRating() is not None and slider.status == STARTED:
            continueRoutine = False
        
        # *text_2* updates
        if text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_2.frameNStart = frameN  # exact frame index
            text_2.tStart = t  # local t and not account for scr refresh
            text_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
            text_2.setAutoDraw(True)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in confidenceComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "confidence"-------
    for thisComponent in confidenceComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    training.addData('slider.response', slider.getRating())
    training.addData('slider.rt', slider.getRT())
    training.addData('slider.started', slider.tStartRefresh)
    training.addData('slider.stopped', slider.tStopRefresh)
    training.addData('text_2.started', text_2.tStartRefresh)
    training.addData('text_2.stopped', text_2.tStopRefresh)
    win.mouseVisible = False
    # the Routine "confidence" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "score"-------
    continueRoutine = True
    routineTimer.add(3.000000)
    # update component parameters for each repeat
    score1.setText(trial_value
)
    Score.reset()
    Score.setText('    Puntuación ganada:\n')
    accumulatedvalue.setText(value_trial)
    accumulated_value.reset()
    accumulated_value.setText('Accumulated score')
    text_9.setText(percentage)
    # keep track of which components have finished
    scoreComponents = [score1, Score, accumulatedvalue, accumulated_value, text_9]
    for thisComponent in scoreComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    scoreClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "score"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = scoreClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=scoreClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *score1* updates
        if score1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            score1.frameNStart = frameN  # exact frame index
            score1.tStart = t  # local t and not account for scr refresh
            score1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(score1, 'tStartRefresh')  # time at next scr refresh
            score1.setAutoDraw(True)
        if score1.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > score1.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                score1.tStop = t  # not accounting for scr refresh
                score1.frameNStop = frameN  # exact frame index
                win.timeOnFlip(score1, 'tStopRefresh')  # time at next scr refresh
                score1.setAutoDraw(False)
        
        # *Score* updates
        if Score.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Score.frameNStart = frameN  # exact frame index
            Score.tStart = t  # local t and not account for scr refresh
            Score.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Score, 'tStartRefresh')  # time at next scr refresh
            Score.setAutoDraw(True)
        if Score.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > Score.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                Score.tStop = t  # not accounting for scr refresh
                Score.frameNStop = frameN  # exact frame index
                win.timeOnFlip(Score, 'tStopRefresh')  # time at next scr refresh
                Score.setAutoDraw(False)
        
        # *accumulatedvalue* updates
        if accumulatedvalue.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            accumulatedvalue.frameNStart = frameN  # exact frame index
            accumulatedvalue.tStart = t  # local t and not account for scr refresh
            accumulatedvalue.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(accumulatedvalue, 'tStartRefresh')  # time at next scr refresh
            accumulatedvalue.setAutoDraw(True)
        if accumulatedvalue.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > accumulatedvalue.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                accumulatedvalue.tStop = t  # not accounting for scr refresh
                accumulatedvalue.frameNStop = frameN  # exact frame index
                win.timeOnFlip(accumulatedvalue, 'tStopRefresh')  # time at next scr refresh
                accumulatedvalue.setAutoDraw(False)
        
        # *accumulated_value* updates
        if accumulated_value.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            accumulated_value.frameNStart = frameN  # exact frame index
            accumulated_value.tStart = t  # local t and not account for scr refresh
            accumulated_value.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(accumulated_value, 'tStartRefresh')  # time at next scr refresh
            accumulated_value.setAutoDraw(True)
        if accumulated_value.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > accumulated_value.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                accumulated_value.tStop = t  # not accounting for scr refresh
                accumulated_value.frameNStop = frameN  # exact frame index
                win.timeOnFlip(accumulated_value, 'tStopRefresh')  # time at next scr refresh
                accumulated_value.setAutoDraw(False)
        
        # *text_9* updates
        if text_9.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_9.frameNStart = frameN  # exact frame index
            text_9.tStart = t  # local t and not account for scr refresh
            text_9.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_9, 'tStartRefresh')  # time at next scr refresh
            text_9.setAutoDraw(True)
        if text_9.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_9.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                text_9.tStop = t  # not accounting for scr refresh
                text_9.frameNStop = frameN  # exact frame index
                win.timeOnFlip(text_9, 'tStopRefresh')  # time at next scr refresh
                text_9.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in scoreComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "score"-------
    for thisComponent in scoreComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    training.addData('score1.started', score1.tStartRefresh)
    training.addData('score1.stopped', score1.tStopRefresh)
    training.addData('Score.started', Score.tStartRefresh)
    training.addData('Score.stopped', Score.tStopRefresh)
    training.addData('accumulatedvalue.started', accumulatedvalue.tStartRefresh)
    training.addData('accumulatedvalue.stopped', accumulatedvalue.tStopRefresh)
    training.addData('accumulated_value.started', accumulated_value.tStartRefresh)
    training.addData('accumulated_value.stopped', accumulated_value.tStopRefresh)
    training.addData('text_9.started', text_9.tStartRefresh)
    training.addData('text_9.stopped', text_9.tStopRefresh)
    
    # ------Prepare to start Routine "inter"-------
    continueRoutine = True
    routineTimer.add(0.100000)
    # update component parameters for each repeat
    # keep track of which components have finished
    interComponents = [image]
    for thisComponent in interComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    interClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "inter"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = interClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=interClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *image* updates
        if image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            image.frameNStart = frameN  # exact frame index
            image.tStart = t  # local t and not account for scr refresh
            image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image, 'tStartRefresh')  # time at next scr refresh
            image.setAutoDraw(True)
        if image.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > image.tStartRefresh + 0.1-frameTolerance:
                # keep track of stop time/frame for later
                image.tStop = t  # not accounting for scr refresh
                image.frameNStop = frameN  # exact frame index
                win.timeOnFlip(image, 'tStopRefresh')  # time at next scr refresh
                image.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in interComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "inter"-------
    for thisComponent in interComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    training.addData('image.started', image.tStartRefresh)
    training.addData('image.stopped', image.tStopRefresh)
    thisExp.nextEntry()
    
# completed 1.0 repeats of 'training'

# get names of stimulus parameters
if training.trialList in ([], [None], None):
    params = []
else:
    params = training.trialList[0].keys()
# save data for this loop
training.saveAsExcel(filename + '.xlsx', sheetName='training',
    stimOut=params,
    dataOut=['n','all_mean','all_std', 'all_raw'])
training.saveAsText(filename + 'training.csv', delim=',',
    stimOut=params,
    dataOut=['n','all_mean','all_std', 'all_raw'])

# ------Prepare to start Routine "starttrial"-------
continueRoutine = True
# update component parameters for each repeat
# setup some python lists for storing info about the mouse_8
gotValidClick = False  # until a click is received
value_trial = 0
# keep track of which components have finished
starttrialComponents = [text_14, mouse_8]
for thisComponent in starttrialComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
starttrialClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "starttrial"-------
while continueRoutine:
    # get current time
    t = starttrialClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=starttrialClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *text_14* updates
    if text_14.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        text_14.frameNStart = frameN  # exact frame index
        text_14.tStart = t  # local t and not account for scr refresh
        text_14.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text_14, 'tStartRefresh')  # time at next scr refresh
        text_14.setAutoDraw(True)
    # *mouse_8* updates
    if mouse_8.status == NOT_STARTED and t >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        mouse_8.frameNStart = frameN  # exact frame index
        mouse_8.tStart = t  # local t and not account for scr refresh
        mouse_8.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(mouse_8, 'tStartRefresh')  # time at next scr refresh
        mouse_8.status = STARTED
        mouse_8.mouseClock.reset()
        prevButtonState = mouse_8.getPressed()  # if button is down already this ISN'T a new click
    if mouse_8.status == STARTED:  # only update if started and not finished!
        buttons = mouse_8.getPressed()
        if buttons != prevButtonState:  # button state changed?
            prevButtonState = buttons
            if sum(buttons) > 0:  # state changed to a new click
                # abort routine on response
                continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in starttrialComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "starttrial"-------
for thisComponent in starttrialComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('text_14.started', text_14.tStartRefresh)
thisExp.addData('text_14.stopped', text_14.tStopRefresh)
# store data for thisExp (ExperimentHandler)
x, y = mouse_8.getPos()
buttons = mouse_8.getPressed()
thisExp.addData('mouse_8.x', x)
thisExp.addData('mouse_8.y', y)
thisExp.addData('mouse_8.leftButton', buttons[0])
thisExp.addData('mouse_8.midButton', buttons[1])
thisExp.addData('mouse_8.rightButton', buttons[2])
thisExp.addData('mouse_8.started', mouse_8.tStart)
thisExp.addData('mouse_8.stopped', mouse_8.tStop)
thisExp.nextEntry()
# the Routine "starttrial" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of conditions etc
trials = data.TrialHandler(nReps=1.0, method='random', 
    extraInfo=expInfo, originPath=-1,
    trialList=data.importConditions('trial11.xlsx'),
    seed=None, name='trials')
thisExp.addLoop(trials)  # add the loop to the experiment
thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
if thisTrial != None:
    for paramName in thisTrial:
        exec('{} = thisTrial[paramName]'.format(paramName))

for thisTrial in trials:
    currentLoop = trials
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            exec('{} = thisTrial[paramName]'.format(paramName))
    
    # ------Prepare to start Routine "trial"-------
    continueRoutine = True
    routineTimer.add(1.000000)
    # update component parameters for each repeat
    fix.setPos((0, 0))
    fix.setSize((0.05, 0.05))
    fix.setOri(0.0)
    #arduino_board.digital[arduino_pin].write(1)
    #arduino_board.pass_time(0.1)
    #arduino_board.digital[arduino_pin].write(0)
    # keep track of which components have finished
    trialComponents = [fix, etRecord_2]
    for thisComponent in trialComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    trialClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "trial"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = trialClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=trialClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *fix* updates
        if fix.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            fix.frameNStart = frameN  # exact frame index
            fix.tStart = t  # local t and not account for scr refresh
            fix.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fix, 'tStartRefresh')  # time at next scr refresh
            fix.setAutoDraw(True)
        if fix.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > fix.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                fix.tStop = t  # not accounting for scr refresh
                fix.frameNStop = frameN  # exact frame index
                win.timeOnFlip(fix, 'tStopRefresh')  # time at next scr refresh
                fix.setAutoDraw(False)
        # *etRecord_2* updates
        if etRecord_2.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            etRecord_2.frameNStart = frameN  # exact frame index
            etRecord_2.tStart = t  # local t and not account for scr refresh
            etRecord_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(etRecord_2, 'tStartRefresh')  # time at next scr refresh
            etRecord_2.status = STARTED
        if etRecord_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > etRecord_2.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                etRecord_2.tStop = t  # not accounting for scr refresh
                etRecord_2.frameNStop = frameN  # exact frame index
                win.timeOnFlip(etRecord_2, 'tStopRefresh')  # time at next scr refresh
                etRecord_2.status = FINISHED
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in trialComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "trial"-------
    for thisComponent in trialComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    trials.addData('fix.started', fix.tStartRefresh)
    trials.addData('fix.stopped', fix.tStopRefresh)
    # make sure the eyetracker recording stops
    if etRecord_2.status != FINISHED:
        etRecord_2.status = FINISHED
    
    # ------Prepare to start Routine "offer1"-------
    continueRoutine = True
    routineTimer.add(2.000000)
    # update component parameters for each repeat
    cl1.setFillColor(colorl)
    cl1.setPos((-0.4, -(0.5-heightl)/2))
    cl1.setSize((0.3, heightl))
    cl1.setLineColor('white')
    # keep track of which components have finished
    offer1Components = [rl1, cl1, etRecord_6]
    for thisComponent in offer1Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    offer1Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "offer1"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = offer1Clock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=offer1Clock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *rl1* updates
        if rl1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            rl1.frameNStart = frameN  # exact frame index
            rl1.tStart = t  # local t and not account for scr refresh
            rl1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(rl1, 'tStartRefresh')  # time at next scr refresh
            rl1.setAutoDraw(True)
        if rl1.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > rl1.tStartRefresh + 2-frameTolerance:
                # keep track of stop time/frame for later
                rl1.tStop = t  # not accounting for scr refresh
                rl1.frameNStop = frameN  # exact frame index
                win.timeOnFlip(rl1, 'tStopRefresh')  # time at next scr refresh
                rl1.setAutoDraw(False)
        
        # *cl1* updates
        if cl1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            cl1.frameNStart = frameN  # exact frame index
            cl1.tStart = t  # local t and not account for scr refresh
            cl1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(cl1, 'tStartRefresh')  # time at next scr refresh
            cl1.setAutoDraw(True)
        if cl1.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > cl1.tStartRefresh + 2-frameTolerance:
                # keep track of stop time/frame for later
                cl1.tStop = t  # not accounting for scr refresh
                cl1.frameNStop = frameN  # exact frame index
                win.timeOnFlip(cl1, 'tStopRefresh')  # time at next scr refresh
                cl1.setAutoDraw(False)
        # *etRecord_6* updates
        if etRecord_6.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            etRecord_6.frameNStart = frameN  # exact frame index
            etRecord_6.tStart = t  # local t and not account for scr refresh
            etRecord_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(etRecord_6, 'tStartRefresh')  # time at next scr refresh
            etRecord_6.status = STARTED
        if etRecord_6.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > etRecord_6.tStartRefresh + 2-frameTolerance:
                # keep track of stop time/frame for later
                etRecord_6.tStop = t  # not accounting for scr refresh
                etRecord_6.frameNStop = frameN  # exact frame index
                win.timeOnFlip(etRecord_6, 'tStopRefresh')  # time at next scr refresh
                etRecord_6.status = FINISHED
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in offer1Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "offer1"-------
    for thisComponent in offer1Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    trials.addData('rl1.started', rl1.tStartRefresh)
    trials.addData('rl1.stopped', rl1.tStopRefresh)
    trials.addData('cl1.started', cl1.tStartRefresh)
    trials.addData('cl1.stopped', cl1.tStopRefresh)
    # make sure the eyetracker recording stops
    if etRecord_6.status != FINISHED:
        etRecord_6.status = FINISHED
    
    # ------Prepare to start Routine "delay1"-------
    continueRoutine = True
    routineTimer.add(2.000000)
    # update component parameters for each repeat
    fix_2.setPos((0, 0))
    fix_2.setSize((0.05, 0.05))
    fix_2.setOri(0.0)
    # keep track of which components have finished
    delay1Components = [image_2, fix_2]
    for thisComponent in delay1Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    delay1Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "delay1"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = delay1Clock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=delay1Clock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *image_2* updates
        if image_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            image_2.frameNStart = frameN  # exact frame index
            image_2.tStart = t  # local t and not account for scr refresh
            image_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_2, 'tStartRefresh')  # time at next scr refresh
            image_2.setAutoDraw(True)
        if image_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > image_2.tStartRefresh + 2-frameTolerance:
                # keep track of stop time/frame for later
                image_2.tStop = t  # not accounting for scr refresh
                image_2.frameNStop = frameN  # exact frame index
                win.timeOnFlip(image_2, 'tStopRefresh')  # time at next scr refresh
                image_2.setAutoDraw(False)
        
        # *fix_2* updates
        if fix_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            fix_2.frameNStart = frameN  # exact frame index
            fix_2.tStart = t  # local t and not account for scr refresh
            fix_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fix_2, 'tStartRefresh')  # time at next scr refresh
            fix_2.setAutoDraw(True)
        if fix_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > fix_2.tStartRefresh + 2-frameTolerance:
                # keep track of stop time/frame for later
                fix_2.tStop = t  # not accounting for scr refresh
                fix_2.frameNStop = frameN  # exact frame index
                win.timeOnFlip(fix_2, 'tStopRefresh')  # time at next scr refresh
                fix_2.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in delay1Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "delay1"-------
    for thisComponent in delay1Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    trials.addData('image_2.started', image_2.tStartRefresh)
    trials.addData('image_2.stopped', image_2.tStopRefresh)
    trials.addData('fix_2.started', fix_2.tStartRefresh)
    trials.addData('fix_2.stopped', fix_2.tStopRefresh)
    
    # ------Prepare to start Routine "offer2"-------
    continueRoutine = True
    routineTimer.add(2.000000)
    # update component parameters for each repeat
    cr1.setFillColor(colorr)
    cr1.setPos((0.4 , -(0.5-heightr)/2))
    cr1.setSize((0.3, heightr))
    cr1.setLineColor('white')
    # keep track of which components have finished
    offer2Components = [rr1, cr1, etRecord_5]
    for thisComponent in offer2Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    offer2Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "offer2"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = offer2Clock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=offer2Clock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *rr1* updates
        if rr1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            rr1.frameNStart = frameN  # exact frame index
            rr1.tStart = t  # local t and not account for scr refresh
            rr1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(rr1, 'tStartRefresh')  # time at next scr refresh
            rr1.setAutoDraw(True)
        if rr1.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > rr1.tStartRefresh + 2-frameTolerance:
                # keep track of stop time/frame for later
                rr1.tStop = t  # not accounting for scr refresh
                rr1.frameNStop = frameN  # exact frame index
                win.timeOnFlip(rr1, 'tStopRefresh')  # time at next scr refresh
                rr1.setAutoDraw(False)
        
        # *cr1* updates
        if cr1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            cr1.frameNStart = frameN  # exact frame index
            cr1.tStart = t  # local t and not account for scr refresh
            cr1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(cr1, 'tStartRefresh')  # time at next scr refresh
            cr1.setAutoDraw(True)
        if cr1.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > cr1.tStartRefresh + 2-frameTolerance:
                # keep track of stop time/frame for later
                cr1.tStop = t  # not accounting for scr refresh
                cr1.frameNStop = frameN  # exact frame index
                win.timeOnFlip(cr1, 'tStopRefresh')  # time at next scr refresh
                cr1.setAutoDraw(False)
        # *etRecord_5* updates
        if etRecord_5.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            etRecord_5.frameNStart = frameN  # exact frame index
            etRecord_5.tStart = t  # local t and not account for scr refresh
            etRecord_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(etRecord_5, 'tStartRefresh')  # time at next scr refresh
            etRecord_5.status = STARTED
        if etRecord_5.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > etRecord_5.tStartRefresh + 2-frameTolerance:
                # keep track of stop time/frame for later
                etRecord_5.tStop = t  # not accounting for scr refresh
                etRecord_5.frameNStop = frameN  # exact frame index
                win.timeOnFlip(etRecord_5, 'tStopRefresh')  # time at next scr refresh
                etRecord_5.status = FINISHED
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in offer2Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "offer2"-------
    for thisComponent in offer2Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    trials.addData('rr1.started', rr1.tStartRefresh)
    trials.addData('rr1.stopped', rr1.tStopRefresh)
    trials.addData('cr1.started', cr1.tStartRefresh)
    trials.addData('cr1.stopped', cr1.tStopRefresh)
    # make sure the eyetracker recording stops
    if etRecord_5.status != FINISHED:
        etRecord_5.status = FINISHED
    
    # ------Prepare to start Routine "delay2"-------
    continueRoutine = True
    routineTimer.add(2.000000)
    # update component parameters for each repeat
    fix_3.setPos((0, 0))
    fix_3.setSize((0.05, 0.05))
    fix_3.setOri(0.0)
    # keep track of which components have finished
    delay2Components = [image_3, fix_3]
    for thisComponent in delay2Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    delay2Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "delay2"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = delay2Clock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=delay2Clock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *image_3* updates
        if image_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            image_3.frameNStart = frameN  # exact frame index
            image_3.tStart = t  # local t and not account for scr refresh
            image_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_3, 'tStartRefresh')  # time at next scr refresh
            image_3.setAutoDraw(True)
        if image_3.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > image_3.tStartRefresh + 2-frameTolerance:
                # keep track of stop time/frame for later
                image_3.tStop = t  # not accounting for scr refresh
                image_3.frameNStop = frameN  # exact frame index
                win.timeOnFlip(image_3, 'tStopRefresh')  # time at next scr refresh
                image_3.setAutoDraw(False)
        
        # *fix_3* updates
        if fix_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            fix_3.frameNStart = frameN  # exact frame index
            fix_3.tStart = t  # local t and not account for scr refresh
            fix_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fix_3, 'tStartRefresh')  # time at next scr refresh
            fix_3.setAutoDraw(True)
        if fix_3.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > fix_3.tStartRefresh + 2-frameTolerance:
                # keep track of stop time/frame for later
                fix_3.tStop = t  # not accounting for scr refresh
                fix_3.frameNStop = frameN  # exact frame index
                win.timeOnFlip(fix_3, 'tStopRefresh')  # time at next scr refresh
                fix_3.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in delay2Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "delay2"-------
    for thisComponent in delay2Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    trials.addData('image_3.started', image_3.tStartRefresh)
    trials.addData('image_3.stopped', image_3.tStopRefresh)
    trials.addData('fix_3.started', fix_3.tStartRefresh)
    trials.addData('fix_3.stopped', fix_3.tStopRefresh)
    
    # ------Prepare to start Routine "observ"-------
    continueRoutine = True
    routineTimer.add(2.000000)
    # update component parameters for each repeat
    rl.setPos((-0.5, 0))
    rl.setSize((0.3, 0.5))
    cl.setFillColor(colorl)
    cl.setPos((-0.5, -(0.5-heightl)/2))
    cl.setSize((0.3, heightl))
    rr.setPos((0.5, 0))
    rr.setSize((0.3, 0.5))
    cr.setFillColor(colorr)
    cr.setPos((0.5 , -(0.5-heightr)/2))
    cr.setSize((0.3, heightr))
    # keep track of which components have finished
    observComponents = [rl, cl, rr, cr, etRecord_3]
    for thisComponent in observComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    observClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "observ"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = observClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=observClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *rl* updates
        if rl.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            rl.frameNStart = frameN  # exact frame index
            rl.tStart = t  # local t and not account for scr refresh
            rl.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(rl, 'tStartRefresh')  # time at next scr refresh
            rl.setAutoDraw(True)
        if rl.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > rl.tStartRefresh + 2-frameTolerance:
                # keep track of stop time/frame for later
                rl.tStop = t  # not accounting for scr refresh
                rl.frameNStop = frameN  # exact frame index
                win.timeOnFlip(rl, 'tStopRefresh')  # time at next scr refresh
                rl.setAutoDraw(False)
        
        # *cl* updates
        if cl.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            cl.frameNStart = frameN  # exact frame index
            cl.tStart = t  # local t and not account for scr refresh
            cl.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(cl, 'tStartRefresh')  # time at next scr refresh
            cl.setAutoDraw(True)
        if cl.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > cl.tStartRefresh + 2-frameTolerance:
                # keep track of stop time/frame for later
                cl.tStop = t  # not accounting for scr refresh
                cl.frameNStop = frameN  # exact frame index
                win.timeOnFlip(cl, 'tStopRefresh')  # time at next scr refresh
                cl.setAutoDraw(False)
        
        # *rr* updates
        if rr.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            rr.frameNStart = frameN  # exact frame index
            rr.tStart = t  # local t and not account for scr refresh
            rr.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(rr, 'tStartRefresh')  # time at next scr refresh
            rr.setAutoDraw(True)
        if rr.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > rr.tStartRefresh + 2-frameTolerance:
                # keep track of stop time/frame for later
                rr.tStop = t  # not accounting for scr refresh
                rr.frameNStop = frameN  # exact frame index
                win.timeOnFlip(rr, 'tStopRefresh')  # time at next scr refresh
                rr.setAutoDraw(False)
        
        # *cr* updates
        if cr.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            cr.frameNStart = frameN  # exact frame index
            cr.tStart = t  # local t and not account for scr refresh
            cr.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(cr, 'tStartRefresh')  # time at next scr refresh
            cr.setAutoDraw(True)
        if cr.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > cr.tStartRefresh + 2-frameTolerance:
                # keep track of stop time/frame for later
                cr.tStop = t  # not accounting for scr refresh
                cr.frameNStop = frameN  # exact frame index
                win.timeOnFlip(cr, 'tStopRefresh')  # time at next scr refresh
                cr.setAutoDraw(False)
        # *etRecord_3* updates
        if etRecord_3.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            etRecord_3.frameNStart = frameN  # exact frame index
            etRecord_3.tStart = t  # local t and not account for scr refresh
            etRecord_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(etRecord_3, 'tStartRefresh')  # time at next scr refresh
            etRecord_3.status = STARTED
        if etRecord_3.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > etRecord_3.tStartRefresh + 2-frameTolerance:
                # keep track of stop time/frame for later
                etRecord_3.tStop = t  # not accounting for scr refresh
                etRecord_3.frameNStop = frameN  # exact frame index
                win.timeOnFlip(etRecord_3, 'tStopRefresh')  # time at next scr refresh
                etRecord_3.status = FINISHED
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in observComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "observ"-------
    for thisComponent in observComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    trials.addData('rl.started', rl.tStartRefresh)
    trials.addData('rl.stopped', rl.tStopRefresh)
    trials.addData('cl.started', cl.tStartRefresh)
    trials.addData('cl.stopped', cl.tStopRefresh)
    trials.addData('rr.started', rr.tStartRefresh)
    trials.addData('rr.stopped', rr.tStopRefresh)
    trials.addData('cr.started', cr.tStartRefresh)
    trials.addData('cr.stopped', cr.tStopRefresh)
    # make sure the eyetracker recording stops
    if etRecord_3.status != FINISHED:
        etRecord_3.status = FINISHED
    
    # ------Prepare to start Routine "decision_trials"-------
    continueRoutine = True
    routineTimer.add(30.000000)
    # update component parameters for each repeat
    decr_2.setFillColor('none')
    decr_2.setOpacity(None)
    decr_2.setContrast(1.0)
    decr_2.setPos((-0.5 ,0))
    decr_2.setSize((0.3, 0.5))
    decr_2.setOri(0.0)
    decr_2.setLineColor('white')
    decr_2.setLineWidth(1.0)
    dec2_2.setPos((0.5 ,0))
    dec2_2.setSize((0.3, 0.5))
    dec2_2.setOri(0.0)
    sound_2.setSound('A', secs=0.5, hamming=True)
    sound_2.setVolume(1.0, log=False)
    # setup some python lists for storing info about the mouse_7
    gotValidClick = False  # until a click is received
    # keep track of which components have finished
    decision_trialsComponents = [decr_2, dec2_2, etRecord_7, sound_2, mouse_7]
    for thisComponent in decision_trialsComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    decision_trialsClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "decision_trials"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = decision_trialsClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=decision_trialsClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *decr_2* updates
        if decr_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            decr_2.frameNStart = frameN  # exact frame index
            decr_2.tStart = t  # local t and not account for scr refresh
            decr_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(decr_2, 'tStartRefresh')  # time at next scr refresh
            decr_2.setAutoDraw(True)
        if decr_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > decr_2.tStartRefresh + 30-frameTolerance:
                # keep track of stop time/frame for later
                decr_2.tStop = t  # not accounting for scr refresh
                decr_2.frameNStop = frameN  # exact frame index
                win.timeOnFlip(decr_2, 'tStopRefresh')  # time at next scr refresh
                decr_2.setAutoDraw(False)
        
        # *dec2_2* updates
        if dec2_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            dec2_2.frameNStart = frameN  # exact frame index
            dec2_2.tStart = t  # local t and not account for scr refresh
            dec2_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(dec2_2, 'tStartRefresh')  # time at next scr refresh
            dec2_2.setAutoDraw(True)
        if dec2_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > dec2_2.tStartRefresh + 30-frameTolerance:
                # keep track of stop time/frame for later
                dec2_2.tStop = t  # not accounting for scr refresh
                dec2_2.frameNStop = frameN  # exact frame index
                win.timeOnFlip(dec2_2, 'tStopRefresh')  # time at next scr refresh
                dec2_2.setAutoDraw(False)
        # *etRecord_7* updates
        if etRecord_7.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            etRecord_7.frameNStart = frameN  # exact frame index
            etRecord_7.tStart = t  # local t and not account for scr refresh
            etRecord_7.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(etRecord_7, 'tStartRefresh')  # time at next scr refresh
            etRecord_7.status = STARTED
        if etRecord_7.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > etRecord_7.tStartRefresh + 30-frameTolerance:
                # keep track of stop time/frame for later
                etRecord_7.tStop = t  # not accounting for scr refresh
                etRecord_7.frameNStop = frameN  # exact frame index
                win.timeOnFlip(etRecord_7, 'tStopRefresh')  # time at next scr refresh
                etRecord_7.status = FINISHED
        # start/stop sound_2
        if sound_2.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
            # keep track of start time/frame for later
            sound_2.frameNStart = frameN  # exact frame index
            sound_2.tStart = t  # local t and not account for scr refresh
            sound_2.tStartRefresh = tThisFlipGlobal  # on global time
            sound_2.play(when=win)  # sync with win flip
        if sound_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > sound_2.tStartRefresh + 0.5-frameTolerance:
                # keep track of stop time/frame for later
                sound_2.tStop = t  # not accounting for scr refresh
                sound_2.frameNStop = frameN  # exact frame index
                win.timeOnFlip(sound_2, 'tStopRefresh')  # time at next scr refresh
                sound_2.stop()
        # *mouse_7* updates
        if mouse_7.status == NOT_STARTED and t >= 2-frameTolerance:
            # keep track of start time/frame for later
            mouse_7.frameNStart = frameN  # exact frame index
            mouse_7.tStart = t  # local t and not account for scr refresh
            mouse_7.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse_7, 'tStartRefresh')  # time at next scr refresh
            mouse_7.status = STARTED
            mouse_7.mouseClock.reset()
            prevButtonState = mouse_7.getPressed()  # if button is down already this ISN'T a new click
        if mouse_7.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > mouse_7.tStartRefresh + 28-frameTolerance:
                # keep track of stop time/frame for later
                mouse_7.tStop = t  # not accounting for scr refresh
                mouse_7.frameNStop = frameN  # exact frame index
                win.timeOnFlip(mouse_7, 'tStopRefresh')  # time at next scr refresh
                mouse_7.status = FINISHED
        if mouse_7.status == STARTED:  # only update if started and not finished!
            buttons = mouse_7.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # abort routine on response
                    continueRoutine = False
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in decision_trialsComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "decision_trials"-------
    for thisComponent in decision_trialsComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    trials.addData('decr_2.started', decr_2.tStartRefresh)
    trials.addData('decr_2.stopped', decr_2.tStopRefresh)
    trials.addData('dec2_2.started', dec2_2.tStartRefresh)
    trials.addData('dec2_2.stopped', dec2_2.tStopRefresh)
    # make sure the eyetracker recording stops
    if etRecord_7.status != FINISHED:
        etRecord_7.status = FINISHED
    color_value={"lightgrey":5, "blue":7,"green":10}
    Er = heightr*2* color_value[colorr];
    El = heightl*2* color_value[colorl]
    import math as mt
    # mouse.rightButton ==1 (means right) .leftButton==1(means left)
    if (mouse.getPressed()[2] ==1 and Er > El) or (mouse.getPressed()[0]==1  and El > Er): 
        true_choice= true_choice+1
    
    
    mouse_l = 0
    mouse_r= 0
    
    if mouse.getPressed()[0] ==1:
        trial_value= np.random.choice([0,1],p=[(1-heightl / 0.5),(heightl / 0.5)])*color_value[colorl]
        text_to_show=trial_value
        mouse_l=1
        value_trial = trial_value+value_trial 
        print("mouse 1")
    elif mouse.getPressed()[2] ==1 :
        trial_value= np.random.choice([0,1],p=[(1-heightr / 0.5),(heightr / 0.5)])*color_value[colorr]
        text_to_show=trial_value
        value_trial  = trial_value+value_trial
        mouse_r=1
        print("mouse 2")
    
    percentage='Ha realizado el '+str(np.round((trials.thisN+1)*100/trials.nTotal))+'% del experimento!'
    
    mouse.setPos([0,1])
    sound_2.stop()  # ensure sound has stopped at end of routine
    trials.addData('sound_2.started', sound_2.tStartRefresh)
    trials.addData('sound_2.stopped', sound_2.tStopRefresh)
    # store data for trials (TrialHandler)
    x, y = mouse_7.getPos()
    buttons = mouse_7.getPressed()
    trials.addData('mouse_7.x', x)
    trials.addData('mouse_7.y', y)
    trials.addData('mouse_7.leftButton', buttons[0])
    trials.addData('mouse_7.midButton', buttons[1])
    trials.addData('mouse_7.rightButton', buttons[2])
    trials.addData('mouse_7.started', mouse_7.tStart)
    trials.addData('mouse_7.stopped', mouse_7.tStop)
    
    # ------Prepare to start Routine "confidence"-------
    continueRoutine = True
    # update component parameters for each repeat
    slider.reset()
    win.mouseVisible = True
    mouse.setPos([0,1])
    # keep track of which components have finished
    confidenceComponents = [slider, text_2]
    for thisComponent in confidenceComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    confidenceClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "confidence"-------
    while continueRoutine:
        # get current time
        t = confidenceClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=confidenceClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *slider* updates
        if slider.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            slider.frameNStart = frameN  # exact frame index
            slider.tStart = t  # local t and not account for scr refresh
            slider.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(slider, 'tStartRefresh')  # time at next scr refresh
            slider.setAutoDraw(True)
        
        # Check slider for response to end routine
        if slider.getRating() is not None and slider.status == STARTED:
            continueRoutine = False
        
        # *text_2* updates
        if text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_2.frameNStart = frameN  # exact frame index
            text_2.tStart = t  # local t and not account for scr refresh
            text_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
            text_2.setAutoDraw(True)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in confidenceComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "confidence"-------
    for thisComponent in confidenceComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    trials.addData('slider.response', slider.getRating())
    trials.addData('slider.rt', slider.getRT())
    trials.addData('slider.started', slider.tStartRefresh)
    trials.addData('slider.stopped', slider.tStopRefresh)
    trials.addData('text_2.started', text_2.tStartRefresh)
    trials.addData('text_2.stopped', text_2.tStopRefresh)
    win.mouseVisible = False
    # the Routine "confidence" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "score"-------
    continueRoutine = True
    routineTimer.add(3.000000)
    # update component parameters for each repeat
    score1.setText(trial_value
)
    Score.reset()
    Score.setText('    Puntuación ganada:\n')
    accumulatedvalue.setText(value_trial)
    accumulated_value.reset()
    accumulated_value.setText('Accumulated score')
    text_9.setText(percentage)
    # keep track of which components have finished
    scoreComponents = [score1, Score, accumulatedvalue, accumulated_value, text_9]
    for thisComponent in scoreComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    scoreClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "score"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = scoreClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=scoreClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *score1* updates
        if score1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            score1.frameNStart = frameN  # exact frame index
            score1.tStart = t  # local t and not account for scr refresh
            score1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(score1, 'tStartRefresh')  # time at next scr refresh
            score1.setAutoDraw(True)
        if score1.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > score1.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                score1.tStop = t  # not accounting for scr refresh
                score1.frameNStop = frameN  # exact frame index
                win.timeOnFlip(score1, 'tStopRefresh')  # time at next scr refresh
                score1.setAutoDraw(False)
        
        # *Score* updates
        if Score.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Score.frameNStart = frameN  # exact frame index
            Score.tStart = t  # local t and not account for scr refresh
            Score.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Score, 'tStartRefresh')  # time at next scr refresh
            Score.setAutoDraw(True)
        if Score.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > Score.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                Score.tStop = t  # not accounting for scr refresh
                Score.frameNStop = frameN  # exact frame index
                win.timeOnFlip(Score, 'tStopRefresh')  # time at next scr refresh
                Score.setAutoDraw(False)
        
        # *accumulatedvalue* updates
        if accumulatedvalue.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            accumulatedvalue.frameNStart = frameN  # exact frame index
            accumulatedvalue.tStart = t  # local t and not account for scr refresh
            accumulatedvalue.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(accumulatedvalue, 'tStartRefresh')  # time at next scr refresh
            accumulatedvalue.setAutoDraw(True)
        if accumulatedvalue.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > accumulatedvalue.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                accumulatedvalue.tStop = t  # not accounting for scr refresh
                accumulatedvalue.frameNStop = frameN  # exact frame index
                win.timeOnFlip(accumulatedvalue, 'tStopRefresh')  # time at next scr refresh
                accumulatedvalue.setAutoDraw(False)
        
        # *accumulated_value* updates
        if accumulated_value.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            accumulated_value.frameNStart = frameN  # exact frame index
            accumulated_value.tStart = t  # local t and not account for scr refresh
            accumulated_value.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(accumulated_value, 'tStartRefresh')  # time at next scr refresh
            accumulated_value.setAutoDraw(True)
        if accumulated_value.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > accumulated_value.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                accumulated_value.tStop = t  # not accounting for scr refresh
                accumulated_value.frameNStop = frameN  # exact frame index
                win.timeOnFlip(accumulated_value, 'tStopRefresh')  # time at next scr refresh
                accumulated_value.setAutoDraw(False)
        
        # *text_9* updates
        if text_9.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_9.frameNStart = frameN  # exact frame index
            text_9.tStart = t  # local t and not account for scr refresh
            text_9.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_9, 'tStartRefresh')  # time at next scr refresh
            text_9.setAutoDraw(True)
        if text_9.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_9.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                text_9.tStop = t  # not accounting for scr refresh
                text_9.frameNStop = frameN  # exact frame index
                win.timeOnFlip(text_9, 'tStopRefresh')  # time at next scr refresh
                text_9.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in scoreComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "score"-------
    for thisComponent in scoreComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    trials.addData('score1.started', score1.tStartRefresh)
    trials.addData('score1.stopped', score1.tStopRefresh)
    trials.addData('Score.started', Score.tStartRefresh)
    trials.addData('Score.stopped', Score.tStopRefresh)
    trials.addData('accumulatedvalue.started', accumulatedvalue.tStartRefresh)
    trials.addData('accumulatedvalue.stopped', accumulatedvalue.tStopRefresh)
    trials.addData('accumulated_value.started', accumulated_value.tStartRefresh)
    trials.addData('accumulated_value.stopped', accumulated_value.tStopRefresh)
    trials.addData('text_9.started', text_9.tStartRefresh)
    trials.addData('text_9.stopped', text_9.tStopRefresh)
    
    # ------Prepare to start Routine "inter"-------
    continueRoutine = True
    routineTimer.add(0.100000)
    # update component parameters for each repeat
    # keep track of which components have finished
    interComponents = [image]
    for thisComponent in interComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    interClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "inter"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = interClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=interClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *image* updates
        if image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            image.frameNStart = frameN  # exact frame index
            image.tStart = t  # local t and not account for scr refresh
            image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image, 'tStartRefresh')  # time at next scr refresh
            image.setAutoDraw(True)
        if image.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > image.tStartRefresh + 0.1-frameTolerance:
                # keep track of stop time/frame for later
                image.tStop = t  # not accounting for scr refresh
                image.frameNStop = frameN  # exact frame index
                win.timeOnFlip(image, 'tStopRefresh')  # time at next scr refresh
                image.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in interComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "inter"-------
    for thisComponent in interComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    trials.addData('image.started', image.tStartRefresh)
    trials.addData('image.stopped', image.tStopRefresh)
    thisExp.nextEntry()
    
# completed 1.0 repeats of 'trials'

# get names of stimulus parameters
if trials.trialList in ([], [None], None):
    params = []
else:
    params = trials.trialList[0].keys()
# save data for this loop
trials.saveAsExcel(filename + '.xlsx', sheetName='trials',
    stimOut=params,
    dataOut=['n','all_mean','all_std', 'all_raw'])
trials.saveAsText(filename + 'trials.csv', delim=',',
    stimOut=params,
    dataOut=['n','all_mean','all_std', 'all_raw'])

# ------Prepare to start Routine "calfinal"-------
continueRoutine = True
routineTimer.add(10.000000)
# update component parameters for each repeat
text_5.setText('Puntuación')
text_8.setText(value_trial)
# keep track of which components have finished
calfinalComponents = [text_5, text_8]
for thisComponent in calfinalComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
calfinalClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "calfinal"-------
while continueRoutine and routineTimer.getTime() > 0:
    # get current time
    t = calfinalClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=calfinalClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *text_5* updates
    if text_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        text_5.frameNStart = frameN  # exact frame index
        text_5.tStart = t  # local t and not account for scr refresh
        text_5.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text_5, 'tStartRefresh')  # time at next scr refresh
        text_5.setAutoDraw(True)
    if text_5.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > text_5.tStartRefresh + 10-frameTolerance:
            # keep track of stop time/frame for later
            text_5.tStop = t  # not accounting for scr refresh
            text_5.frameNStop = frameN  # exact frame index
            win.timeOnFlip(text_5, 'tStopRefresh')  # time at next scr refresh
            text_5.setAutoDraw(False)
    
    # *text_8* updates
    if text_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        text_8.frameNStart = frameN  # exact frame index
        text_8.tStart = t  # local t and not account for scr refresh
        text_8.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text_8, 'tStartRefresh')  # time at next scr refresh
        text_8.setAutoDraw(True)
    if text_8.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > text_8.tStartRefresh + 10-frameTolerance:
            # keep track of stop time/frame for later
            text_8.tStop = t  # not accounting for scr refresh
            text_8.frameNStop = frameN  # exact frame index
            win.timeOnFlip(text_8, 'tStopRefresh')  # time at next scr refresh
            text_8.setAutoDraw(False)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in calfinalComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "calfinal"-------
for thisComponent in calfinalComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('text_5.started', text_5.tStartRefresh)
thisExp.addData('text_5.stopped', text_5.tStopRefresh)
thisExp.addData('text_8.started', text_8.tStartRefresh)
thisExp.addData('text_8.stopped', text_8.tStopRefresh)

# Flip one final time so any remaining win.callOnFlip() 
# and win.timeOnFlip() tasks get executed before quitting
win.flip()

# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsWideText(filename+'.csv', delim='auto')
thisExp.saveAsPickle(filename)
logging.flush()
# make sure everything is closed down
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()
