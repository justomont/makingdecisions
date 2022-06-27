#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2021.2.3),
    on Fri May 20 11:08:36 2022
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

import psychopy.iohub as io
from psychopy.hardware import keyboard
from psychopy.hardware import emotiv

from pyfirmata import Arduino
arduino_pin = (7)

arduino_board = Arduino('/dev/cu.usbmodem11101')

trial_counter = 0 


# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Store info about the experiment session
psychopyVersion = '2021.2.3'
expName = 'ContextChoice'  # from the Builder filename that created this script
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
    originPath='/Volumes/GoogleDrive/Mi unidad/_BRAIN+COGNITION/0_TFM/Experiments/ContextChoice_ESP/ContextChoice.py',
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
    size=[1920, 1080], fullscr=True, screen=0, 
    winType='pyglet', allowGUI=False, allowStencil=False,
    monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
    blendMode='avg', useFBO=True, 
    units='height')
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess

# Setup eyetracking
ioDevice = 'eyetracker.hw.tobii.EyeTracker'
ioConfig = {
    ioDevice: {
        'name': 'tracker',
        'model_name': '',
        'serial_number': '',
        'runtime_settings': {
            'sampling_rate': 120.0,
        }
    }
}
ioSession = '1'
if 'session' in expInfo:
    ioSession = str(expInfo['session'])
ioServer = io.launchHubServer(window=win, experiment_code='ContextChoice', session_code=ioSession, datastore_name=filename, **ioConfig)
eyetracker = ioServer.getDevice('tracker')

# create a default keyboard (e.g. to check for escape)
defaultKeyboard = keyboard.Keyboard()

# Initialize components for Routine "cal_instructions"
cal_instructionsClock = core.Clock()
calibration_txt = visual.TextStim(win=win, name='calibration_txt',
    text='La posición de su mirada se detecterá ahora mismo. Para hacerlo aparecerán unos puntos verdes en la pantalla. Debe seguirlos con la mirada en todo momento. Cuando se paren, debe fijar la mirada en ellos. \n\nPulse cualquier tecla en el ratón para continuar. ',
    font='Open Sans',
    pos=(0, 0), height=0.04, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
mouse_5 = event.Mouse(win=win)
x, y = [None, None]
mouse_5.mouseClock = core.Clock()
win.mouseVisible = False

# Initialize components for Routine "context_instructions"
context_instructionsClock = core.Clock()
Disclaimer_2 = visual.TextStim(win=win, name='Disclaimer_2',
    text='Está a punto de realizar un ejercicio de toma de decisiones. En cada intento se le presentará una pregunta o una situación y un par de opciones representadas con imágenes, una a la derecha y otra a la izquierda. \n\nPulse cualquier tecla en el ratón para continuar.',
    font='Open Sans',
    pos=(0, 0), height=0.035, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
mouse = event.Mouse(win=win)
x, y = [None, None]
mouse.mouseClock = core.Clock()
win.mouseVisible = False

# Initialize components for Routine "FP_instructions"
FP_instructionsClock = core.Clock()
Disclaimer_3 = visual.TextStim(win=win, name='Disclaimer_3',
    text='Antes de que se presenten las imágenes, un + como este aparecerá en la pantalla:\n\n\n\n\n\n\nDebe mirarlo fijamente hasta que desaparezca.\n\nPulse cualquier tecla en el ratón para continuar.',
    font='Open Sans',
    pos=(0, 0), height=0.035, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
FixationCross_2 = visual.ShapeStim(
    win=win, name='FixationCross_2', vertices='cross',
    size=(0.05, 0.05),
    ori=0.0, pos=(0, 0),
    lineWidth=0.6,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-1.0, interpolate=True)
mouse_2 = event.Mouse(win=win)
x, y = [None, None]
mouse_2.mouseClock = core.Clock()

# Initialize components for Routine "decision_instructions"
decision_instructionsClock = core.Clock()
Disclaimer_4 = visual.TextStim(win=win, name='Disclaimer_4',
    text='No podrá elegir una opción al principio. Cuando oiga un sonido, podrá informar de su decisión haciendo click en la tecla izquierda del ratón si ha elegido la opción a la izquierda, o haciendo click en el botón derecho del ratón si elige la opción mostrada a la derecha. \n\nPulse cualquier tecla en el ratón para continuar.',
    font='Open Sans',
    pos=(0, 0), height=0.035, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
mouse_3 = event.Mouse(win=win)
x, y = [None, None]
mouse_3.mouseClock = core.Clock()

# Initialize components for Routine "instructions"
instructionsClock = core.Clock()
Disclaimer = visual.TextStim(win=win, name='Disclaimer',
    text='En algunos intentos, las imágenes desaparecerán antes de que se emita el sonido que le permite tomar una decisión. En otros intentos las imágenes siempre estarán presentes.  \n\nDespués de cada decisión se le harán 3 preguntas. \n\nPulse cualquier tecla en el ratón para continuar.',
    font='Open Sans',
    pos=(0, 0), height=0.035, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
mouse_4 = event.Mouse(win=win)
x, y = [None, None]
mouse_4.mouseClock = core.Clock()

# Initialize components for Routine "example_intro"
example_introClock = core.Clock()
example_instr_2 = visual.TextStim(win=win, name='example_instr_2',
    text='Ahora verá y participará en un ejemplo.\n\nPulse cualquier tecla en el ratón para continuar.',
    font='Open Sans',
    pos=(0, 0), height=0.035, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
mouse_7 = event.Mouse(win=win)
x, y = [None, None]
mouse_7.mouseClock = core.Clock()

# Initialize components for Routine "trial_example"
trial_exampleClock = core.Clock()
Context_ex = visual.TextStim(win=win, name='Context_ex',
    text='',
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
FC_ex = visual.ShapeStim(
    win=win, name='FC_ex', vertices='cross',
    size=(0.05, 0.05),
    ori=0.0, pos=(0, 0),
    lineWidth=0.6,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-1.0, interpolate=True)
stimA_ex = visual.ImageStim(
    win=win,
    name='stimA_ex', 
    image='sin', mask=None,
    ori=0.0, pos=(-0.5, 0), size=(0.25, 0.25),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-2.0)
stimB_ex = visual.ImageStim(
    win=win,
    name='stimB_ex', 
    image='sin', mask=None,
    ori=0.0, pos=(0.5, 0), size=(0.25, 0.25),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-3.0)
sound_3 = sound.Sound('A', secs=0.5, stereo=True, hamming=True,
    name='sound_3')
sound_3.setVolume(1.0)
resp_Ex = event.Mouse(win=win)
x, y = [None, None]
resp_Ex.mouseClock = core.Clock()
tip_ex = visual.TextStim(win=win, name='tip_ex',
    text='Ahora puede indicar que opción prefiere haciendo click en el botón derecho del ratón si ha elegido la opción a la derecha, o haciendo click en el botón izquierdo del ratón si elige la opción a la izquierda.',
    font='Open Sans',
    pos=(0, 0.25), height=0.035, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-6.0);
stimA_ex_2 = visual.ImageStim(
    win=win,
    name='stimA_ex_2', 
    image='sin', mask=None,
    ori=0.0, pos=(-0.5, 0), size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-7.0)
stimB_ex_2 = visual.ImageStim(
    win=win,
    name='stimB_ex_2', 
    image='sin', mask=None,
    ori=0.0, pos=(0.5, 0), size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-8.0)
squareA_ex = visual.Rect(
    win=win, name='squareA_ex',
    width=(0.25, 0.25)[0], height=(0.25, 0.25)[1],
    ori=0.0, pos=(-0.5, 0),
    lineWidth=5.0,     colorSpace='rgb',  lineColor='lime', fillColor=None,
    opacity=None, depth=-9.0, interpolate=True)
squareB_ex = visual.Rect(
    win=win, name='squareB_ex',
    width=(0.25, 0.25)[0], height=(0.25, 0.25)[1],
    ori=0.0, pos=(0.5, 0),
    lineWidth=5.0,     colorSpace='rgb',  lineColor='lime', fillColor=None,
    opacity=None, depth=-10.0, interpolate=True)

# Initialize components for Routine "assessment_ex"
assessment_exClock = core.Clock()
question_confidence_2 = visual.TextStim(win=win, name='question_confidence_2',
    text='',
    font='Open Sans',
    pos=(0, 0.1), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
slider_confidence_2 = visual.Slider(win=win, name='slider_confidence_2',
    startValue=None, size=(1.1, 0.1), pos=(0, 0.0), units=None,
    labels=("Nada","Absolutamente"), ticks=(0,1, 2, 3, 4, 5,6,7,8,9,10), granularity=1.0,
    style='radio', styleTweaks=(), opacity=None,
    color='white', fillColor='Red', borderColor='ghostwhite', colorSpace='rgb',
    font='Open Sans', labelHeight=0.05,
    flip=False, depth=-1, readOnly=False)
tip_ex_2 = visual.TextStim(win=win, name='tip_ex_2',
    text='Aquí debe hacer click en el nivel de confianza en si mismo/a que siente en relación a la decisión. ',
    font='Open Sans',
    pos=(0, -0.25), height=0.035, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-3.0);

# Initialize components for Routine "valueA_ex"
valueA_exClock = core.Clock()
question_valueA_2 = visual.TextStim(win=win, name='question_valueA_2',
    text='Dada la siguiente pregunta/situación, ¿cómo de probable es que elija la opción que se presenta debajo?',
    font='Open Sans',
    pos=(0, 0.3), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
slider_valueA_2 = visual.Slider(win=win, name='slider_valueA_2',
    startValue=None, size=(1, 0.05), pos=(0, -0.35), units=None,
    labels=("-","+"), ticks=(-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1, 2, 3, 4, 5,6,7,8,9,10), granularity=1.0,
    style='radio', styleTweaks=(), opacity=None,
    color='white', fillColor='Red', borderColor='ghostwhite', colorSpace='rgb',
    font='Open Sans', labelHeight=0.05,
    flip=False, depth=-1, readOnly=False)
stimA_4 = visual.ImageStim(
    win=win,
    name='stimA_4', 
    image='sin', mask=None,
    ori=0.0, pos=(0, -0.15), size=(0.25, 0.25),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-2.0)
Context_valueA_ex = visual.TextStim(win=win, name='Context_valueA_ex',
    text='',
    font='Open Sans',
    pos=(0, 0.15), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-3.0);

# Initialize components for Routine "valueB_ex"
valueB_exClock = core.Clock()
question_valueB_2 = visual.TextStim(win=win, name='question_valueB_2',
    text='Dada la siguiente pregunta/situación, ¿cómo de probable es que elija la opción que se presenta debajo?',
    font='Open Sans',
    pos=(0, 0.3), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
slider_valueB_2 = visual.Slider(win=win, name='slider_valueB_2',
    startValue=None, size=(1.0, 0.05), pos=(0, -0.35), units=None,
    labels=("-","+"), ticks=(-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1, 2, 3, 4, 5,6,7,8,9,10), granularity=1.0,
    style='radio', styleTweaks=(), opacity=None,
    color='white', fillColor='Red', borderColor='ghostwhite', colorSpace='rgb',
    font='Open Sans', labelHeight=0.05,
    flip=False, depth=-1, readOnly=False)
stimB_4 = visual.ImageStim(
    win=win,
    name='stimB_4', 
    image='sin', mask=None,
    ori=0.0, pos=(0, -0.15), size=(0.25, 0.25),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-2.0)
Context_valueB_ex = visual.TextStim(win=win, name='Context_valueB_ex',
    text='',
    font='Open Sans',
    pos=(0, 0.15), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-4.0);

# Initialize components for Routine "intro"
introClock = core.Clock()
Disclaimer_5 = visual.TextStim(win=win, name='Disclaimer_5',
    text='¡Enhorabuena! Ha terminado el entrenamiento, ¡está a punto de empezar el experimento!\n\nSi aún tiene alguna duda, es el momento de preguntarla. \n\nSi está preparado/a para empezar, haga click en cualquier botón del ratón.',
    font='Open Sans',
    pos=(0, 0), height=0.035, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
mouse_6 = event.Mouse(win=win)
x, y = [None, None]
mouse_6.mouseClock = core.Clock()

# Initialize components for Routine "trial"
trialClock = core.Clock()
ContextPresentation = visual.TextStim(win=win, name='ContextPresentation',
    text='',
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
FixationCross = visual.ShapeStim(
    win=win, name='FixationCross', vertices='cross',
    size=(0.05, 0.05),
    ori=0.0, pos=(0, 0),
    lineWidth=0.6,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-1.0, interpolate=True)
stimA = visual.ImageStim(
    win=win,
    name='stimA', 
    image='sin', mask=None,
    ori=0.0, pos=(-0.5, 0), size=(0.25, 0.25),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-2.0)
stimB = visual.ImageStim(
    win=win,
    name='stimB', 
    image='sin', mask=None,
    ori=0.0, pos=(0.5, 0), size=(0.25, 0.25),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-3.0)
squareA = visual.Rect(
    win=win, name='squareA',
    width=(0.25, 0.25)[0], height=(0.25, 0.25)[1],
    ori=0.0, pos=(-0.5, 0),
    lineWidth=5.0,     colorSpace='rgb',  lineColor='lime', fillColor=None,
    opacity=None, depth=-4.0, interpolate=True)
squareB = visual.Rect(
    win=win, name='squareB',
    width=(0.25, 0.25)[0], height=(0.25, 0.25)[1],
    ori=0.0, pos=(0.5, 0),
    lineWidth=5.0,     colorSpace='rgb',  lineColor='lime', fillColor=None,
    opacity=None, depth=-5.0, interpolate=True)
recEYE_S = hardware.eyetracker.EyetrackerControl(
    server=ioServer,
    tracker=eyetracker
)

# Initialize components for Routine "decision"
decisionClock = core.Clock()
stimA_3 = visual.ImageStim(
    win=win,
    name='stimA_3', 
    image='sin', mask=None,
    ori=0.0, pos=(-0.5, 0), size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=0.0)
stimB_3 = visual.ImageStim(
    win=win,
    name='stimB_3', 
    image='sin', mask=None,
    ori=0.0, pos=(0.5, 0), size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-1.0)
squareA_5 = visual.Rect(
    win=win, name='squareA_5',
    width=(0.25, 0.25)[0], height=(0.25, 0.25)[1],
    ori=0.0, pos=(-0.5, 0),
    lineWidth=5.0,     colorSpace='rgb',  lineColor='lime', fillColor=None,
    opacity=None, depth=-2.0, interpolate=True)
squareB_3 = visual.Rect(
    win=win, name='squareB_3',
    width=(0.25, 0.25)[0], height=(0.25, 0.25)[1],
    ori=0.0, pos=(0.5, 0),
    lineWidth=5.0,     colorSpace='rgb',  lineColor='lime', fillColor=None,
    opacity=None, depth=-3.0, interpolate=True)
resp_choice = event.Mouse(win=win)
x, y = [None, None]
resp_choice.mouseClock = core.Clock()
sound_1 = sound.Sound('A', secs=0.5, stereo=True, hamming=True,
    name='sound_1')
sound_1.setVolume(1.0)
recEYE = hardware.eyetracker.EyetrackerControl(
    server=ioServer,
    tracker=eyetracker
)

# Initialize components for Routine "confidence"
confidenceClock = core.Clock()
question_confidence = visual.TextStim(win=win, name='question_confidence',
    text='¿Cómo de seguro/a se siente de su decisión? ',
    font='Open Sans',
    pos=(0, 0.1), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
slider_confidence = visual.Slider(win=win, name='slider_confidence',
    startValue=None, size=(1.1, 0.1), pos=(0, 0.0), units=None,
    labels=("Nada","Absolutamente"), ticks=(0,1, 2, 3, 4, 5,6,7,8,9,10), granularity=1.0,
    style='radio', styleTweaks=(), opacity=None,
    color='white', fillColor='Red', borderColor='ghostwhite', colorSpace='rgb',
    font='Open Sans', labelHeight=0.05,
    flip=False, depth=-1, readOnly=False)
win.mouseVisible = False

# Initialize components for Routine "valueA"
valueAClock = core.Clock()
question_valueA = visual.TextStim(win=win, name='question_valueA',
    text='Dada la siguiente pregunta/situación, ¿cómo de probable es que elija la opción que se presenta debajo?',
    font='Open Sans',
    pos=(0, 0.3), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
slider_valueA = visual.Slider(win=win, name='slider_valueA',
    startValue=None, size=(1, 0.05), pos=(0, -0.35), units=None,
    labels=("-","+"), ticks=(-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1, 2, 3, 4, 5,6,7,8,9,10), granularity=1.0,
    style='radio', styleTweaks=(), opacity=None,
    color='white', fillColor='Red', borderColor='ghostwhite', colorSpace='rgb',
    font='Open Sans', labelHeight=0.05,
    flip=False, depth=-1, readOnly=False)
stimA_2 = visual.ImageStim(
    win=win,
    name='stimA_2', 
    image='sin', mask=None,
    ori=0.0, pos=(0, -0.15), size=(0.25, 0.25),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-2.0)
context_valueA = visual.TextStim(win=win, name='context_valueA',
    text='',
    font='Open Sans',
    pos=(0, 0.15), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-3.0);

# Initialize components for Routine "valueB"
valueBClock = core.Clock()
question_valueB = visual.TextStim(win=win, name='question_valueB',
    text='Dada la siguiente pregunta/situación, ¿cómo de probable es que elija la opción que se presenta debajo?',
    font='Open Sans',
    pos=(0, 0.3), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
slider_valueB = visual.Slider(win=win, name='slider_valueB',
    startValue=None, size=(1.0, 0.05), pos=(0, -0.35), units=None,
    labels=("-","+"), ticks=(-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1, 2, 3, 4, 5,6,7,8,9,10), granularity=1.0,
    style='radio', styleTweaks=(), opacity=None,
    color='white', fillColor='Red', borderColor='ghostwhite', colorSpace='rgb',
    font='Open Sans', labelHeight=0.05,
    flip=False, depth=-1, readOnly=False)
stimB_2 = visual.ImageStim(
    win=win,
    name='stimB_2', 
    image='sin', mask=None,
    ori=0.0, pos=(0, -0.15), size=(0.25, 0.25),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-2.0)
Context_valueB = visual.TextStim(win=win, name='Context_valueB',
    text='',
    font='Open Sans',
    pos=(0, 0.15), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-4.0);

# Initialize components for Routine "percentage_trials"
percentage_trialsClock = core.Clock()
info_percentage = visual.TextStim(win=win, name='info_percentage',
    text='',
    font='Open Sans',
    pos=(0, 0), height=0.035, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-1.0);

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine 

# ------Prepare to start Routine "cal_instructions"-------
continueRoutine = True
# update component parameters for each repeat
# setup some python lists for storing info about the mouse_5
gotValidClick = False  # until a click is received
win.mouseVisible = False
# keep track of which components have finished
cal_instructionsComponents = [calibration_txt, mouse_5]
for thisComponent in cal_instructionsComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
cal_instructionsClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "cal_instructions"-------
while continueRoutine:
    # get current time
    t = cal_instructionsClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=cal_instructionsClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *calibration_txt* updates
    if calibration_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        calibration_txt.frameNStart = frameN  # exact frame index
        calibration_txt.tStart = t  # local t and not account for scr refresh
        calibration_txt.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(calibration_txt, 'tStartRefresh')  # time at next scr refresh
        calibration_txt.setAutoDraw(True)
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
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in cal_instructionsComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "cal_instructions"-------
for thisComponent in cal_instructionsComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('calibration_txt.started', calibration_txt.tStartRefresh)
thisExp.addData('calibration_txt.stopped', calibration_txt.tStopRefresh)
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
# the Routine "cal_instructions" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# -------Run Routine 'calibration'-------

# define target for calibration
calibrationTarget = visual.TargetStim(win, 
    name='calibrationTarget',
    radius=0.01, fillColor='ghostwhite', borderColor='white', lineWidth=2.0,
    innerRadius=0.0035, innerFillColor='green', innerBorderColor='white', innerLineWidth=2.0,
    colorSpace='rgb', units=None
)
# define parameters for calibration
calibration = hardware.eyetracker.EyetrackerCalibration(win, 
    eyetracker, calibrationTarget,
    units=None, colorSpace='rgb',
    progressMode='time', targetDur=1.5, expandScale=1.5,
    targetLayout='NINE_POINTS', randomisePos=True,
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
    radius=0.01, fillColor='ghostwhite', borderColor='white', lineWidth=2.0,
    innerRadius=0.0035, innerFillColor='green', innerBorderColor='white', innerLineWidth=2.0,
    colorSpace='rgb', units=None
)
# define parameters for validation
validation = iohub.ValidationProcedure(win,
    target=validationTarget,
    gaze_cursor='green', 
    positions='NINE_POINTS', randomize_positions=True,
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

# ------Prepare to start Routine "context_instructions"-------
continueRoutine = True
# update component parameters for each repeat
# setup some python lists for storing info about the mouse
gotValidClick = False  # until a click is received
win.mouseVisible = False
# keep track of which components have finished
context_instructionsComponents = [Disclaimer_2, mouse]
for thisComponent in context_instructionsComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
context_instructionsClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "context_instructions"-------
while continueRoutine:
    # get current time
    t = context_instructionsClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=context_instructionsClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *Disclaimer_2* updates
    if Disclaimer_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        Disclaimer_2.frameNStart = frameN  # exact frame index
        Disclaimer_2.tStart = t  # local t and not account for scr refresh
        Disclaimer_2.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(Disclaimer_2, 'tStartRefresh')  # time at next scr refresh
        Disclaimer_2.setAutoDraw(True)
    # *mouse* updates
    if mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        mouse.frameNStart = frameN  # exact frame index
        mouse.tStart = t  # local t and not account for scr refresh
        mouse.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(mouse, 'tStartRefresh')  # time at next scr refresh
        mouse.status = STARTED
        mouse.mouseClock.reset()
        prevButtonState = mouse.getPressed()  # if button is down already this ISN'T a new click
    if mouse.status == STARTED:  # only update if started and not finished!
        buttons = mouse.getPressed()
        if buttons != prevButtonState:  # button state changed?
            prevButtonState = buttons
            if sum(buttons) > 0:  # state changed to a new click
                # abort routine on response
                continueRoutine = False
    win.mouseVisible = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in context_instructionsComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "context_instructions"-------
for thisComponent in context_instructionsComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('Disclaimer_2.started', Disclaimer_2.tStartRefresh)
thisExp.addData('Disclaimer_2.stopped', Disclaimer_2.tStopRefresh)
# store data for thisExp (ExperimentHandler)
x, y = mouse.getPos()
buttons = mouse.getPressed()
thisExp.addData('mouse.x', x)
thisExp.addData('mouse.y', y)
thisExp.addData('mouse.leftButton', buttons[0])
thisExp.addData('mouse.midButton', buttons[1])
thisExp.addData('mouse.rightButton', buttons[2])
thisExp.addData('mouse.started', mouse.tStart)
thisExp.addData('mouse.stopped', mouse.tStop)
thisExp.nextEntry()
# the Routine "context_instructions" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "FP_instructions"-------
continueRoutine = True
# update component parameters for each repeat
# setup some python lists for storing info about the mouse_2
gotValidClick = False  # until a click is received
# keep track of which components have finished
FP_instructionsComponents = [Disclaimer_3, FixationCross_2, mouse_2]
for thisComponent in FP_instructionsComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
FP_instructionsClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "FP_instructions"-------
while continueRoutine:
    # get current time
    t = FP_instructionsClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=FP_instructionsClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *Disclaimer_3* updates
    if Disclaimer_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        Disclaimer_3.frameNStart = frameN  # exact frame index
        Disclaimer_3.tStart = t  # local t and not account for scr refresh
        Disclaimer_3.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(Disclaimer_3, 'tStartRefresh')  # time at next scr refresh
        Disclaimer_3.setAutoDraw(True)
    
    # *FixationCross_2* updates
    if FixationCross_2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
        # keep track of start time/frame for later
        FixationCross_2.frameNStart = frameN  # exact frame index
        FixationCross_2.tStart = t  # local t and not account for scr refresh
        FixationCross_2.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(FixationCross_2, 'tStartRefresh')  # time at next scr refresh
        FixationCross_2.setAutoDraw(True)
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
    for thisComponent in FP_instructionsComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "FP_instructions"-------
for thisComponent in FP_instructionsComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('Disclaimer_3.started', Disclaimer_3.tStartRefresh)
thisExp.addData('Disclaimer_3.stopped', Disclaimer_3.tStopRefresh)
thisExp.addData('FixationCross_2.started', FixationCross_2.tStartRefresh)
thisExp.addData('FixationCross_2.stopped', FixationCross_2.tStopRefresh)
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
# the Routine "FP_instructions" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "decision_instructions"-------
continueRoutine = True
# update component parameters for each repeat
# setup some python lists for storing info about the mouse_3
gotValidClick = False  # until a click is received
# keep track of which components have finished
decision_instructionsComponents = [Disclaimer_4, mouse_3]
for thisComponent in decision_instructionsComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
decision_instructionsClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "decision_instructions"-------
while continueRoutine:
    # get current time
    t = decision_instructionsClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=decision_instructionsClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *Disclaimer_4* updates
    if Disclaimer_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        Disclaimer_4.frameNStart = frameN  # exact frame index
        Disclaimer_4.tStart = t  # local t and not account for scr refresh
        Disclaimer_4.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(Disclaimer_4, 'tStartRefresh')  # time at next scr refresh
        Disclaimer_4.setAutoDraw(True)
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
    for thisComponent in decision_instructionsComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "decision_instructions"-------
for thisComponent in decision_instructionsComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('Disclaimer_4.started', Disclaimer_4.tStartRefresh)
thisExp.addData('Disclaimer_4.stopped', Disclaimer_4.tStopRefresh)
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
# the Routine "decision_instructions" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "instructions"-------
continueRoutine = True
# update component parameters for each repeat
# setup some python lists for storing info about the mouse_4
gotValidClick = False  # until a click is received
# keep track of which components have finished
instructionsComponents = [Disclaimer, mouse_4]
for thisComponent in instructionsComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
instructionsClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "instructions"-------
while continueRoutine:
    # get current time
    t = instructionsClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=instructionsClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *Disclaimer* updates
    if Disclaimer.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        Disclaimer.frameNStart = frameN  # exact frame index
        Disclaimer.tStart = t  # local t and not account for scr refresh
        Disclaimer.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(Disclaimer, 'tStartRefresh')  # time at next scr refresh
        Disclaimer.setAutoDraw(True)
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
    for thisComponent in instructionsComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "instructions"-------
for thisComponent in instructionsComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('Disclaimer.started', Disclaimer.tStartRefresh)
thisExp.addData('Disclaimer.stopped', Disclaimer.tStopRefresh)
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
# the Routine "instructions" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "example_intro"-------
continueRoutine = True
# update component parameters for each repeat
# setup some python lists for storing info about the mouse_7
gotValidClick = False  # until a click is received
# keep track of which components have finished
example_introComponents = [example_instr_2, mouse_7]
for thisComponent in example_introComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
example_introClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "example_intro"-------
while continueRoutine:
    # get current time
    t = example_introClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=example_introClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *example_instr_2* updates
    if example_instr_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        example_instr_2.frameNStart = frameN  # exact frame index
        example_instr_2.tStart = t  # local t and not account for scr refresh
        example_instr_2.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(example_instr_2, 'tStartRefresh')  # time at next scr refresh
        example_instr_2.setAutoDraw(True)
    # *mouse_7* updates
    if mouse_7.status == NOT_STARTED and t >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        mouse_7.frameNStart = frameN  # exact frame index
        mouse_7.tStart = t  # local t and not account for scr refresh
        mouse_7.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(mouse_7, 'tStartRefresh')  # time at next scr refresh
        mouse_7.status = STARTED
        mouse_7.mouseClock.reset()
        prevButtonState = mouse_7.getPressed()  # if button is down already this ISN'T a new click
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
    for thisComponent in example_introComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "example_intro"-------
for thisComponent in example_introComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('example_instr_2.started', example_instr_2.tStartRefresh)
thisExp.addData('example_instr_2.stopped', example_instr_2.tStopRefresh)
# store data for thisExp (ExperimentHandler)
x, y = mouse_7.getPos()
buttons = mouse_7.getPressed()
thisExp.addData('mouse_7.x', x)
thisExp.addData('mouse_7.y', y)
thisExp.addData('mouse_7.leftButton', buttons[0])
thisExp.addData('mouse_7.midButton', buttons[1])
thisExp.addData('mouse_7.rightButton', buttons[2])
thisExp.addData('mouse_7.started', mouse_7.tStart)
thisExp.addData('mouse_7.stopped', mouse_7.tStop)
thisExp.nextEntry()
# the Routine "example_intro" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of conditions etc
trials_2 = data.TrialHandler(nReps=1.0, method='sequential', 
    extraInfo=expInfo, originPath=-1,
    trialList=data.importConditions('train_loop.xlsx', selection='0:1'),
    seed=None, name='trials_2')
thisExp.addLoop(trials_2)  # add the loop to the experiment
thisTrial_2 = trials_2.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisTrial_2.rgb)
if thisTrial_2 != None:
    for paramName in thisTrial_2:
        exec('{} = thisTrial_2[paramName]'.format(paramName))

for thisTrial_2 in trials_2:
    currentLoop = trials_2
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_2.rgb)
    if thisTrial_2 != None:
        for paramName in thisTrial_2:
            exec('{} = thisTrial_2[paramName]'.format(paramName))
    
    # ------Prepare to start Routine "trial_example"-------
    continueRoutine = True
    # update component parameters for each repeat
    Context_ex.setText(written_context)
    stimA_ex.setImage(imageA)
    stimB_ex.setImage(imageB)
    sound_3.setSound('A', secs=0.5, hamming=True)
    sound_3.setVolume(1.0, log=False)
    # setup some python lists for storing info about the resp_Ex
    gotValidClick = False  # until a click is received
    stimA_ex_2.setSize((0.25*LaN, 0.25*LaN))
    stimA_ex_2.setImage(imageA)
    stimB_ex_2.setSize((0.25*LaN, 0.25*LaN))
    stimB_ex_2.setImage(imageB)
    # keep track of which components have finished
    trial_exampleComponents = [Context_ex, FC_ex, stimA_ex, stimB_ex, sound_3, resp_Ex, tip_ex, stimA_ex_2, stimB_ex_2, squareA_ex, squareB_ex]
    for thisComponent in trial_exampleComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    trial_exampleClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "trial_example"-------
    while continueRoutine:
        # get current time
        t = trial_exampleClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=trial_exampleClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Context_ex* updates
        if Context_ex.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            Context_ex.frameNStart = frameN  # exact frame index
            Context_ex.tStart = t  # local t and not account for scr refresh
            Context_ex.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Context_ex, 'tStartRefresh')  # time at next scr refresh
            Context_ex.setAutoDraw(True)
        if Context_ex.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > Context_ex.tStartRefresh + 4.0-frameTolerance:
                # keep track of stop time/frame for later
                Context_ex.tStop = t  # not accounting for scr refresh
                Context_ex.frameNStop = frameN  # exact frame index
                win.timeOnFlip(Context_ex, 'tStopRefresh')  # time at next scr refresh
                Context_ex.setAutoDraw(False)
        
        # *FC_ex* updates
        if FC_ex.status == NOT_STARTED and tThisFlip >= 4-frameTolerance:
            # keep track of start time/frame for later
            FC_ex.frameNStart = frameN  # exact frame index
            FC_ex.tStart = t  # local t and not account for scr refresh
            FC_ex.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(FC_ex, 'tStartRefresh')  # time at next scr refresh
            FC_ex.setAutoDraw(True)
        if FC_ex.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > FC_ex.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                FC_ex.tStop = t  # not accounting for scr refresh
                FC_ex.frameNStop = frameN  # exact frame index
                win.timeOnFlip(FC_ex, 'tStopRefresh')  # time at next scr refresh
                FC_ex.setAutoDraw(False)
        
        # *stimA_ex* updates
        if stimA_ex.status == NOT_STARTED and tThisFlip >= 5-frameTolerance:
            # keep track of start time/frame for later
            stimA_ex.frameNStart = frameN  # exact frame index
            stimA_ex.tStart = t  # local t and not account for scr refresh
            stimA_ex.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(stimA_ex, 'tStartRefresh')  # time at next scr refresh
            stimA_ex.setAutoDraw(True)
        if stimA_ex.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > stimA_ex.tStartRefresh + 2-frameTolerance:
                # keep track of stop time/frame for later
                stimA_ex.tStop = t  # not accounting for scr refresh
                stimA_ex.frameNStop = frameN  # exact frame index
                win.timeOnFlip(stimA_ex, 'tStopRefresh')  # time at next scr refresh
                stimA_ex.setAutoDraw(False)
        
        # *stimB_ex* updates
        if stimB_ex.status == NOT_STARTED and tThisFlip >= 5-frameTolerance:
            # keep track of start time/frame for later
            stimB_ex.frameNStart = frameN  # exact frame index
            stimB_ex.tStart = t  # local t and not account for scr refresh
            stimB_ex.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(stimB_ex, 'tStartRefresh')  # time at next scr refresh
            stimB_ex.setAutoDraw(True)
        if stimB_ex.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > stimB_ex.tStartRefresh + 2-frameTolerance:
                # keep track of stop time/frame for later
                stimB_ex.tStop = t  # not accounting for scr refresh
                stimB_ex.frameNStop = frameN  # exact frame index
                win.timeOnFlip(stimB_ex, 'tStopRefresh')  # time at next scr refresh
                stimB_ex.setAutoDraw(False)
        # start/stop sound_3
        if sound_3.status == NOT_STARTED and tThisFlip >= 9-frameTolerance:
            # keep track of start time/frame for later
            sound_3.frameNStart = frameN  # exact frame index
            sound_3.tStart = t  # local t and not account for scr refresh
            sound_3.tStartRefresh = tThisFlipGlobal  # on global time
            sound_3.play(when=win)  # sync with win flip
        if sound_3.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > sound_3.tStartRefresh + 0.5-frameTolerance:
                # keep track of stop time/frame for later
                sound_3.tStop = t  # not accounting for scr refresh
                sound_3.frameNStop = frameN  # exact frame index
                win.timeOnFlip(sound_3, 'tStopRefresh')  # time at next scr refresh
                sound_3.stop()
        # *resp_Ex* updates
        if resp_Ex.status == NOT_STARTED and t >= 9-frameTolerance:
            # keep track of start time/frame for later
            resp_Ex.frameNStart = frameN  # exact frame index
            resp_Ex.tStart = t  # local t and not account for scr refresh
            resp_Ex.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(resp_Ex, 'tStartRefresh')  # time at next scr refresh
            resp_Ex.status = STARTED
            resp_Ex.mouseClock.reset()
            prevButtonState = resp_Ex.getPressed()  # if button is down already this ISN'T a new click
        if resp_Ex.status == STARTED:  # only update if started and not finished!
            buttons = resp_Ex.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    continueRoutine = False        
        # *tip_ex* updates
        if tip_ex.status == NOT_STARTED and tThisFlip >= 9-frameTolerance:
            # keep track of start time/frame for later
            tip_ex.frameNStart = frameN  # exact frame index
            tip_ex.tStart = t  # local t and not account for scr refresh
            tip_ex.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(tip_ex, 'tStartRefresh')  # time at next scr refresh
            tip_ex.setAutoDraw(True)
        
        # *stimA_ex_2* updates
        if stimA_ex_2.status == NOT_STARTED and tThisFlip >= 7-frameTolerance:
            # keep track of start time/frame for later
            stimA_ex_2.frameNStart = frameN  # exact frame index
            stimA_ex_2.tStart = t  # local t and not account for scr refresh
            stimA_ex_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(stimA_ex_2, 'tStartRefresh')  # time at next scr refresh
            stimA_ex_2.setAutoDraw(True)
        
        # *stimB_ex_2* updates
        if stimB_ex_2.status == NOT_STARTED and tThisFlip >= 7-frameTolerance:
            # keep track of start time/frame for later
            stimB_ex_2.frameNStart = frameN  # exact frame index
            stimB_ex_2.tStart = t  # local t and not account for scr refresh
            stimB_ex_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(stimB_ex_2, 'tStartRefresh')  # time at next scr refresh
            stimB_ex_2.setAutoDraw(True)
        
        # *squareA_ex* updates
        if squareA_ex.status == NOT_STARTED and tThisFlip >= 5-frameTolerance:
            # keep track of start time/frame for later
            squareA_ex.frameNStart = frameN  # exact frame index
            squareA_ex.tStart = t  # local t and not account for scr refresh
            squareA_ex.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(squareA_ex, 'tStartRefresh')  # time at next scr refresh
            squareA_ex.setAutoDraw(True)
        
        # *squareB_ex* updates
        if squareB_ex.status == NOT_STARTED and tThisFlip >= 5-frameTolerance:
            # keep track of start time/frame for later
            squareB_ex.frameNStart = frameN  # exact frame index
            squareB_ex.tStart = t  # local t and not account for scr refresh
            squareB_ex.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(squareB_ex, 'tStartRefresh')  # time at next scr refresh
            squareB_ex.setAutoDraw(True)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in trial_exampleComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "trial_example"-------
    for thisComponent in trial_exampleComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    sound_3.stop()  # ensure sound has stopped at end of routine
    # store data for trials_2 (TrialHandler)
    # the Routine "trial_example" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "assessment_ex"-------
    continueRoutine = True
    # update component parameters for each repeat
    question_confidence_2.setText(written_context)
    slider_confidence_2.reset()
    win.mouseVisible = True
    mouse.setPos([0,1])
    # keep track of which components have finished
    assessment_exComponents = [question_confidence_2, slider_confidence_2, tip_ex_2]
    for thisComponent in assessment_exComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    assessment_exClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "assessment_ex"-------
    while continueRoutine:
        # get current time
        t = assessment_exClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=assessment_exClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *question_confidence_2* updates
        if question_confidence_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            question_confidence_2.frameNStart = frameN  # exact frame index
            question_confidence_2.tStart = t  # local t and not account for scr refresh
            question_confidence_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(question_confidence_2, 'tStartRefresh')  # time at next scr refresh
            question_confidence_2.setAutoDraw(True)
        
        # *slider_confidence_2* updates
        if slider_confidence_2.status == NOT_STARTED and tThisFlip >= 0.1-frameTolerance:
            # keep track of start time/frame for later
            slider_confidence_2.frameNStart = frameN  # exact frame index
            slider_confidence_2.tStart = t  # local t and not account for scr refresh
            slider_confidence_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(slider_confidence_2, 'tStartRefresh')  # time at next scr refresh
            slider_confidence_2.setAutoDraw(True)
        
        # Check slider_confidence_2 for response to end routine
        if slider_confidence_2.getRating() is not None and slider_confidence_2.status == STARTED:
            continueRoutine = False
        
        # *tip_ex_2* updates
        if tip_ex_2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            tip_ex_2.frameNStart = frameN  # exact frame index
            tip_ex_2.tStart = t  # local t and not account for scr refresh
            tip_ex_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(tip_ex_2, 'tStartRefresh')  # time at next scr refresh
            tip_ex_2.setAutoDraw(True)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in assessment_exComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "assessment_ex"-------
    for thisComponent in assessment_exComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # the Routine "assessment_ex" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "valueA_ex"-------
    continueRoutine = True
    # update component parameters for each repeat
    slider_valueA_2.reset()
    stimA_4.setImage(imageA)
    Context_valueA_ex.setText(written_context)
    # keep track of which components have finished
    valueA_exComponents = [question_valueA_2, slider_valueA_2, stimA_4, Context_valueA_ex]
    for thisComponent in valueA_exComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    valueA_exClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "valueA_ex"-------
    while continueRoutine:
        # get current time
        t = valueA_exClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=valueA_exClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *question_valueA_2* updates
        if question_valueA_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            question_valueA_2.frameNStart = frameN  # exact frame index
            question_valueA_2.tStart = t  # local t and not account for scr refresh
            question_valueA_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(question_valueA_2, 'tStartRefresh')  # time at next scr refresh
            question_valueA_2.setAutoDraw(True)
        
        # *slider_valueA_2* updates
        if slider_valueA_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            slider_valueA_2.frameNStart = frameN  # exact frame index
            slider_valueA_2.tStart = t  # local t and not account for scr refresh
            slider_valueA_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(slider_valueA_2, 'tStartRefresh')  # time at next scr refresh
            slider_valueA_2.setAutoDraw(True)
        
        # Check slider_valueA_2 for response to end routine
        if slider_valueA_2.getRating() is not None and slider_valueA_2.status == STARTED:
            continueRoutine = False
        
        # *stimA_4* updates
        if stimA_4.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            stimA_4.frameNStart = frameN  # exact frame index
            stimA_4.tStart = t  # local t and not account for scr refresh
            stimA_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(stimA_4, 'tStartRefresh')  # time at next scr refresh
            stimA_4.setAutoDraw(True)
        
        # *Context_valueA_ex* updates
        if Context_valueA_ex.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            Context_valueA_ex.frameNStart = frameN  # exact frame index
            Context_valueA_ex.tStart = t  # local t and not account for scr refresh
            Context_valueA_ex.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Context_valueA_ex, 'tStartRefresh')  # time at next scr refresh
            Context_valueA_ex.setAutoDraw(True)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in valueA_exComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "valueA_ex"-------
    for thisComponent in valueA_exComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    trials_2.addData('question_valueA_2.started', question_valueA_2.tStartRefresh)
    trials_2.addData('question_valueA_2.stopped', question_valueA_2.tStopRefresh)
    trials_2.addData('stimA_4.started', stimA_4.tStartRefresh)
    trials_2.addData('stimA_4.stopped', stimA_4.tStopRefresh)
    trials_2.addData('Context_valueA_ex.started', Context_valueA_ex.tStartRefresh)
    trials_2.addData('Context_valueA_ex.stopped', Context_valueA_ex.tStopRefresh)
    # the Routine "valueA_ex" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "valueB_ex"-------
    continueRoutine = True
    # update component parameters for each repeat
    slider_valueB_2.reset()
    stimB_4.setImage(imageB)
    Context_valueB_ex.setText(written_context)
    # keep track of which components have finished
    valueB_exComponents = [question_valueB_2, slider_valueB_2, stimB_4, Context_valueB_ex]
    for thisComponent in valueB_exComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    valueB_exClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "valueB_ex"-------
    while continueRoutine:
        # get current time
        t = valueB_exClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=valueB_exClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *question_valueB_2* updates
        if question_valueB_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            question_valueB_2.frameNStart = frameN  # exact frame index
            question_valueB_2.tStart = t  # local t and not account for scr refresh
            question_valueB_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(question_valueB_2, 'tStartRefresh')  # time at next scr refresh
            question_valueB_2.setAutoDraw(True)
        
        # *slider_valueB_2* updates
        if slider_valueB_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            slider_valueB_2.frameNStart = frameN  # exact frame index
            slider_valueB_2.tStart = t  # local t and not account for scr refresh
            slider_valueB_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(slider_valueB_2, 'tStartRefresh')  # time at next scr refresh
            slider_valueB_2.setAutoDraw(True)
        
        # Check slider_valueB_2 for response to end routine
        if slider_valueB_2.getRating() is not None and slider_valueB_2.status == STARTED:
            continueRoutine = False
        
        # *stimB_4* updates
        if stimB_4.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            stimB_4.frameNStart = frameN  # exact frame index
            stimB_4.tStart = t  # local t and not account for scr refresh
            stimB_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(stimB_4, 'tStartRefresh')  # time at next scr refresh
            stimB_4.setAutoDraw(True)
        
        # *Context_valueB_ex* updates
        if Context_valueB_ex.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            Context_valueB_ex.frameNStart = frameN  # exact frame index
            Context_valueB_ex.tStart = t  # local t and not account for scr refresh
            Context_valueB_ex.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Context_valueB_ex, 'tStartRefresh')  # time at next scr refresh
            Context_valueB_ex.setAutoDraw(True)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in valueB_exComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "valueB_ex"-------
    for thisComponent in valueB_exComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    win.mouseVisible = False
    # the Routine "valueB_ex" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    thisExp.nextEntry()
    
# completed 1.0 repeats of 'trials_2'

# get names of stimulus parameters
if trials_2.trialList in ([], [None], None):
    params = []
else:
    params = trials_2.trialList[0].keys()
# save data for this loop
trials_2.saveAsExcel(filename + '.xlsx', sheetName='trials_2',
    stimOut=params,
    dataOut=['n','all_mean','all_std', 'all_raw'])

# ------Prepare to start Routine "intro"-------
continueRoutine = True
# update component parameters for each repeat
# setup some python lists for storing info about the mouse_6
gotValidClick = False  # until a click is received
# keep track of which components have finished
introComponents = [Disclaimer_5, mouse_6]
for thisComponent in introComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
introClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "intro"-------
while continueRoutine:
    # get current time
    t = introClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=introClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *Disclaimer_5* updates
    if Disclaimer_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        Disclaimer_5.frameNStart = frameN  # exact frame index
        Disclaimer_5.tStart = t  # local t and not account for scr refresh
        Disclaimer_5.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(Disclaimer_5, 'tStartRefresh')  # time at next scr refresh
        Disclaimer_5.setAutoDraw(True)
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
    for thisComponent in introComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "intro"-------
for thisComponent in introComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('Disclaimer_5.started', Disclaimer_5.tStartRefresh)
thisExp.addData('Disclaimer_5.stopped', Disclaimer_5.tStopRefresh)
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
# the Routine "intro" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of conditions etc
trials = data.TrialHandler(nReps=1.0, method='random', 
    extraInfo=expInfo, originPath=-1,
    trialList=data.importConditions('160Loop.xlsx', selection='0:79'),
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
    routineTimer.add(8.000000)
    # update component parameters for each repeat
    ContextPresentation.setText(written_context)
    stimA.setImage(imageA)
    stimB.setImage(imageB)
    arduino_board.digital[arduino_pin].write(1)
    arduino_board.pass_time(0.1)
    arduino_board.digital[arduino_pin].write(0)
    trial_counter = trial_counter + 1
    # keep track of which components have finished
    trialComponents = [ContextPresentation, FixationCross, stimA, stimB, squareA, squareB, recEYE_S]
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
        
        # *ContextPresentation* updates
        if ContextPresentation.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            ContextPresentation.frameNStart = frameN  # exact frame index
            ContextPresentation.tStart = t  # local t and not account for scr refresh
            ContextPresentation.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(ContextPresentation, 'tStartRefresh')  # time at next scr refresh
            ContextPresentation.setAutoDraw(True)
        if ContextPresentation.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > ContextPresentation.tStartRefresh + 4.0-frameTolerance:
                # keep track of stop time/frame for later
                ContextPresentation.tStop = t  # not accounting for scr refresh
                ContextPresentation.frameNStop = frameN  # exact frame index
                win.timeOnFlip(ContextPresentation, 'tStopRefresh')  # time at next scr refresh
                ContextPresentation.setAutoDraw(False)
        
        # *FixationCross* updates
        if FixationCross.status == NOT_STARTED and tThisFlip >= 4-frameTolerance:
            # keep track of start time/frame for later
            FixationCross.frameNStart = frameN  # exact frame index
            FixationCross.tStart = t  # local t and not account for scr refresh
            FixationCross.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(FixationCross, 'tStartRefresh')  # time at next scr refresh
            FixationCross.setAutoDraw(True)
        if FixationCross.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > FixationCross.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                FixationCross.tStop = t  # not accounting for scr refresh
                FixationCross.frameNStop = frameN  # exact frame index
                win.timeOnFlip(FixationCross, 'tStopRefresh')  # time at next scr refresh
                FixationCross.setAutoDraw(False)
        
        # *stimA* updates
        if stimA.status == NOT_STARTED and tThisFlip >= 5-frameTolerance:
            # keep track of start time/frame for later
            stimA.frameNStart = frameN  # exact frame index
            stimA.tStart = t  # local t and not account for scr refresh
            stimA.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(stimA, 'tStartRefresh')  # time at next scr refresh
            stimA.setAutoDraw(True)
        if stimA.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > stimA.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                stimA.tStop = t  # not accounting for scr refresh
                stimA.frameNStop = frameN  # exact frame index
                win.timeOnFlip(stimA, 'tStopRefresh')  # time at next scr refresh
                stimA.setAutoDraw(False)
        
        # *stimB* updates
        if stimB.status == NOT_STARTED and tThisFlip >= 5-frameTolerance:
            # keep track of start time/frame for later
            stimB.frameNStart = frameN  # exact frame index
            stimB.tStart = t  # local t and not account for scr refresh
            stimB.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(stimB, 'tStartRefresh')  # time at next scr refresh
            stimB.setAutoDraw(True)
        if stimB.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > stimB.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                stimB.tStop = t  # not accounting for scr refresh
                stimB.frameNStop = frameN  # exact frame index
                win.timeOnFlip(stimB, 'tStopRefresh')  # time at next scr refresh
                stimB.setAutoDraw(False)
        
        # *squareA* updates
        if squareA.status == NOT_STARTED and tThisFlip >= 5-frameTolerance:
            # keep track of start time/frame for later
            squareA.frameNStart = frameN  # exact frame index
            squareA.tStart = t  # local t and not account for scr refresh
            squareA.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(squareA, 'tStartRefresh')  # time at next scr refresh
            squareA.setAutoDraw(True)
        if squareA.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > squareA.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                squareA.tStop = t  # not accounting for scr refresh
                squareA.frameNStop = frameN  # exact frame index
                win.timeOnFlip(squareA, 'tStopRefresh')  # time at next scr refresh
                squareA.setAutoDraw(False)
        
        # *squareB* updates
        if squareB.status == NOT_STARTED and tThisFlip >= 5-frameTolerance:
            # keep track of start time/frame for later
            squareB.frameNStart = frameN  # exact frame index
            squareB.tStart = t  # local t and not account for scr refresh
            squareB.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(squareB, 'tStartRefresh')  # time at next scr refresh
            squareB.setAutoDraw(True)
        if squareB.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > squareB.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                squareB.tStop = t  # not accounting for scr refresh
                squareB.frameNStop = frameN  # exact frame index
                win.timeOnFlip(squareB, 'tStopRefresh')  # time at next scr refresh
                squareB.setAutoDraw(False)
        # *recEYE_S* updates
        if recEYE_S.status == NOT_STARTED and t >= 4-frameTolerance:
            # keep track of start time/frame for later
            recEYE_S.frameNStart = frameN  # exact frame index
            recEYE_S.tStart = t  # local t and not account for scr refresh
            recEYE_S.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(recEYE_S, 'tStartRefresh')  # time at next scr refresh
            recEYE_S.status = STARTED
        if recEYE_S.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 8-frameTolerance:
                # keep track of stop time/frame for later
                recEYE_S.tStop = t  # not accounting for scr refresh
                recEYE_S.frameNStop = frameN  # exact frame index
                win.timeOnFlip(recEYE_S, 'tStopRefresh')  # time at next scr refresh
                recEYE_S.status = FINISHED
        
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
    trials.addData('ContextPresentation.started', ContextPresentation.tStartRefresh)
    trials.addData('ContextPresentation.stopped', ContextPresentation.tStopRefresh)
    trials.addData('FixationCross.started', FixationCross.tStartRefresh)
    trials.addData('FixationCross.stopped', FixationCross.tStopRefresh)
    trials.addData('stimA.started', stimA.tStartRefresh)
    trials.addData('stimA.stopped', stimA.tStopRefresh)
    trials.addData('stimB.started', stimB.tStartRefresh)
    trials.addData('stimB.stopped', stimB.tStopRefresh)
    trials.addData('squareA.started', squareA.tStartRefresh)
    trials.addData('squareA.stopped', squareA.tStopRefresh)
    trials.addData('squareB.started', squareB.tStartRefresh)
    trials.addData('squareB.stopped', squareB.tStopRefresh)
    # make sure the eyetracker recording stops
    if recEYE_S.status != FINISHED:
        recEYE_S.status = FINISHED
    
    # ------Prepare to start Routine "decision"-------
    continueRoutine = True
    routineTimer.add(30.000000)
    # update component parameters for each repeat
    stimA_3.setImage(imageA)
    stimB_3.setImage(imageB)
    # setup some python lists for storing info about the resp_choice
    gotValidClick = False  # until a click is received
    sound_1.setSound('A', secs=0.5, hamming=True)
    sound_1.setVolume(1.0, log=False)
    # keep track of which components have finished
    decisionComponents = [stimA_3, stimB_3, squareA_5, squareB_3, resp_choice, sound_1, recEYE]
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
        
        # *stimA_3* updates
        if stimA_3.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            stimA_3.frameNStart = frameN  # exact frame index
            stimA_3.tStart = t  # local t and not account for scr refresh
            stimA_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(stimA_3, 'tStartRefresh')  # time at next scr refresh
            stimA_3.setAutoDraw(True)
        if stimA_3.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > stimA_3.tStartRefresh + 30-frameTolerance:
                # keep track of stop time/frame for later
                stimA_3.tStop = t  # not accounting for scr refresh
                stimA_3.frameNStop = frameN  # exact frame index
                win.timeOnFlip(stimA_3, 'tStopRefresh')  # time at next scr refresh
                stimA_3.setAutoDraw(False)
        if stimA_3.status == STARTED:  # only update if drawing
            stimA_3.setSize((0.25*LaN, 0.25*LaN), log=False)
        
        # *stimB_3* updates
        if stimB_3.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            stimB_3.frameNStart = frameN  # exact frame index
            stimB_3.tStart = t  # local t and not account for scr refresh
            stimB_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(stimB_3, 'tStartRefresh')  # time at next scr refresh
            stimB_3.setAutoDraw(True)
        if stimB_3.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > stimB_3.tStartRefresh + 30-frameTolerance:
                # keep track of stop time/frame for later
                stimB_3.tStop = t  # not accounting for scr refresh
                stimB_3.frameNStop = frameN  # exact frame index
                win.timeOnFlip(stimB_3, 'tStopRefresh')  # time at next scr refresh
                stimB_3.setAutoDraw(False)
        if stimB_3.status == STARTED:  # only update if drawing
            stimB_3.setSize((0.25*LaN, 0.25*LaN), log=False)
        
        # *squareA_5* updates
        if squareA_5.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            squareA_5.frameNStart = frameN  # exact frame index
            squareA_5.tStart = t  # local t and not account for scr refresh
            squareA_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(squareA_5, 'tStartRefresh')  # time at next scr refresh
            squareA_5.setAutoDraw(True)
        if squareA_5.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > squareA_5.tStartRefresh + 30-frameTolerance:
                # keep track of stop time/frame for later
                squareA_5.tStop = t  # not accounting for scr refresh
                squareA_5.frameNStop = frameN  # exact frame index
                win.timeOnFlip(squareA_5, 'tStopRefresh')  # time at next scr refresh
                squareA_5.setAutoDraw(False)
        
        # *squareB_3* updates
        if squareB_3.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            squareB_3.frameNStart = frameN  # exact frame index
            squareB_3.tStart = t  # local t and not account for scr refresh
            squareB_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(squareB_3, 'tStartRefresh')  # time at next scr refresh
            squareB_3.setAutoDraw(True)
        if squareB_3.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > squareB_3.tStartRefresh + 30-frameTolerance:
                # keep track of stop time/frame for later
                squareB_3.tStop = t  # not accounting for scr refresh
                squareB_3.frameNStop = frameN  # exact frame index
                win.timeOnFlip(squareB_3, 'tStopRefresh')  # time at next scr refresh
                squareB_3.setAutoDraw(False)
        # *resp_choice* updates
        if resp_choice.status == NOT_STARTED and t >= 3-frameTolerance:
            # keep track of start time/frame for later
            resp_choice.frameNStart = frameN  # exact frame index
            resp_choice.tStart = t  # local t and not account for scr refresh
            resp_choice.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(resp_choice, 'tStartRefresh')  # time at next scr refresh
            resp_choice.status = STARTED
            resp_choice.mouseClock.reset()
            prevButtonState = resp_choice.getPressed()  # if button is down already this ISN'T a new click
        if resp_choice.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > resp_choice.tStartRefresh + 25-frameTolerance:
                # keep track of stop time/frame for later
                resp_choice.tStop = t  # not accounting for scr refresh
                resp_choice.frameNStop = frameN  # exact frame index
                win.timeOnFlip(resp_choice, 'tStopRefresh')  # time at next scr refresh
                resp_choice.status = FINISHED
        if resp_choice.status == STARTED:  # only update if started and not finished!
            buttons = resp_choice.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # abort routine on response
                    continueRoutine = False
        # start/stop sound_1
        if sound_1.status == NOT_STARTED and tThisFlip >= 3-frameTolerance:
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
        # *recEYE* updates
        if recEYE.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            recEYE.frameNStart = frameN  # exact frame index
            recEYE.tStart = t  # local t and not account for scr refresh
            recEYE.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(recEYE, 'tStartRefresh')  # time at next scr refresh
            recEYE.status = STARTED
        if recEYE.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 30-frameTolerance:
                # keep track of stop time/frame for later
                recEYE.tStop = t  # not accounting for scr refresh
                recEYE.frameNStop = frameN  # exact frame index
                win.timeOnFlip(recEYE, 'tStopRefresh')  # time at next scr refresh
                recEYE.status = FINISHED
        
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
    trials.addData('stimA_3.started', stimA_3.tStartRefresh)
    trials.addData('stimA_3.stopped', stimA_3.tStopRefresh)
    trials.addData('stimB_3.started', stimB_3.tStartRefresh)
    trials.addData('stimB_3.stopped', stimB_3.tStopRefresh)
    trials.addData('squareA_5.started', squareA_5.tStartRefresh)
    trials.addData('squareA_5.stopped', squareA_5.tStopRefresh)
    trials.addData('squareB_3.started', squareB_3.tStartRefresh)
    trials.addData('squareB_3.stopped', squareB_3.tStopRefresh)
    # store data for trials (TrialHandler)
    x, y = resp_choice.getPos()
    buttons = resp_choice.getPressed()
    trials.addData('resp_choice.x', x)
    trials.addData('resp_choice.y', y)
    trials.addData('resp_choice.leftButton', buttons[0])
    trials.addData('resp_choice.midButton', buttons[1])
    trials.addData('resp_choice.rightButton', buttons[2])
    trials.addData('resp_choice.started', resp_choice.tStart)
    trials.addData('resp_choice.stopped', resp_choice.tStop)
    sound_1.stop()  # ensure sound has stopped at end of routine
    trials.addData('sound_1.started', sound_1.tStartRefresh)
    trials.addData('sound_1.stopped', sound_1.tStopRefresh)
    # make sure the eyetracker recording stops
    if recEYE.status != FINISHED:
        recEYE.status = FINISHED
    
    # ------Prepare to start Routine "confidence"-------
    continueRoutine = True
    # update component parameters for each repeat
    slider_confidence.reset()
    win.mouseVisible = True
    mouse.setPos([0,1])
    
    # keep track of which components have finished
    confidenceComponents = [question_confidence, slider_confidence]
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
        
        # *question_confidence* updates
        if question_confidence.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            question_confidence.frameNStart = frameN  # exact frame index
            question_confidence.tStart = t  # local t and not account for scr refresh
            question_confidence.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(question_confidence, 'tStartRefresh')  # time at next scr refresh
            question_confidence.setAutoDraw(True)
        
        # *slider_confidence* updates
        if slider_confidence.status == NOT_STARTED and tThisFlip >= 0.1-frameTolerance:
            # keep track of start time/frame for later
            slider_confidence.frameNStart = frameN  # exact frame index
            slider_confidence.tStart = t  # local t and not account for scr refresh
            slider_confidence.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(slider_confidence, 'tStartRefresh')  # time at next scr refresh
            slider_confidence.setAutoDraw(True)
        
        # Check slider_confidence for response to end routine
        if slider_confidence.getRating() is not None and slider_confidence.status == STARTED:
            continueRoutine = False
        win.mouseVisible = True
        mouse.setPos([0,1])
        
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
    trials.addData('question_confidence.started', question_confidence.tStartRefresh)
    trials.addData('question_confidence.stopped', question_confidence.tStopRefresh)
    trials.addData('slider_confidence.response', slider_confidence.getRating())
    trials.addData('slider_confidence.rt', slider_confidence.getRT())
    trials.addData('slider_confidence.started', slider_confidence.tStartRefresh)
    trials.addData('slider_confidence.stopped', slider_confidence.tStopRefresh)
    # the Routine "confidence" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "valueA"-------
    continueRoutine = True
    # update component parameters for each repeat
    slider_valueA.reset()
    stimA_2.setImage(imageA)
    context_valueA.setText(written_context)
    # keep track of which components have finished
    valueAComponents = [question_valueA, slider_valueA, stimA_2, context_valueA]
    for thisComponent in valueAComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    valueAClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "valueA"-------
    while continueRoutine:
        # get current time
        t = valueAClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=valueAClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *question_valueA* updates
        if question_valueA.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            question_valueA.frameNStart = frameN  # exact frame index
            question_valueA.tStart = t  # local t and not account for scr refresh
            question_valueA.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(question_valueA, 'tStartRefresh')  # time at next scr refresh
            question_valueA.setAutoDraw(True)
        
        # *slider_valueA* updates
        if slider_valueA.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            slider_valueA.frameNStart = frameN  # exact frame index
            slider_valueA.tStart = t  # local t and not account for scr refresh
            slider_valueA.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(slider_valueA, 'tStartRefresh')  # time at next scr refresh
            slider_valueA.setAutoDraw(True)
        
        # Check slider_valueA for response to end routine
        if slider_valueA.getRating() is not None and slider_valueA.status == STARTED:
            continueRoutine = False
        
        # *stimA_2* updates
        if stimA_2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            stimA_2.frameNStart = frameN  # exact frame index
            stimA_2.tStart = t  # local t and not account for scr refresh
            stimA_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(stimA_2, 'tStartRefresh')  # time at next scr refresh
            stimA_2.setAutoDraw(True)
        
        # *context_valueA* updates
        if context_valueA.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            context_valueA.frameNStart = frameN  # exact frame index
            context_valueA.tStart = t  # local t and not account for scr refresh
            context_valueA.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(context_valueA, 'tStartRefresh')  # time at next scr refresh
            context_valueA.setAutoDraw(True)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in valueAComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "valueA"-------
    for thisComponent in valueAComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    trials.addData('question_valueA.started', question_valueA.tStartRefresh)
    trials.addData('question_valueA.stopped', question_valueA.tStopRefresh)
    trials.addData('slider_valueA.response', slider_valueA.getRating())
    trials.addData('stimA_2.started', stimA_2.tStartRefresh)
    trials.addData('stimA_2.stopped', stimA_2.tStopRefresh)
    trials.addData('context_valueA.started', context_valueA.tStartRefresh)
    trials.addData('context_valueA.stopped', context_valueA.tStopRefresh)
    # the Routine "valueA" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "valueB"-------
    continueRoutine = True
    # update component parameters for each repeat
    slider_valueB.reset()
    stimB_2.setImage(imageB)
    Context_valueB.setText(written_context)
    # keep track of which components have finished
    valueBComponents = [question_valueB, slider_valueB, stimB_2, Context_valueB]
    for thisComponent in valueBComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    valueBClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "valueB"-------
    while continueRoutine:
        # get current time
        t = valueBClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=valueBClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *question_valueB* updates
        if question_valueB.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            question_valueB.frameNStart = frameN  # exact frame index
            question_valueB.tStart = t  # local t and not account for scr refresh
            question_valueB.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(question_valueB, 'tStartRefresh')  # time at next scr refresh
            question_valueB.setAutoDraw(True)
        
        # *slider_valueB* updates
        if slider_valueB.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            slider_valueB.frameNStart = frameN  # exact frame index
            slider_valueB.tStart = t  # local t and not account for scr refresh
            slider_valueB.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(slider_valueB, 'tStartRefresh')  # time at next scr refresh
            slider_valueB.setAutoDraw(True)
        
        # Check slider_valueB for response to end routine
        if slider_valueB.getRating() is not None and slider_valueB.status == STARTED:
            continueRoutine = False
        
        # *stimB_2* updates
        if stimB_2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            stimB_2.frameNStart = frameN  # exact frame index
            stimB_2.tStart = t  # local t and not account for scr refresh
            stimB_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(stimB_2, 'tStartRefresh')  # time at next scr refresh
            stimB_2.setAutoDraw(True)
        
        # *Context_valueB* updates
        if Context_valueB.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            Context_valueB.frameNStart = frameN  # exact frame index
            Context_valueB.tStart = t  # local t and not account for scr refresh
            Context_valueB.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Context_valueB, 'tStartRefresh')  # time at next scr refresh
            Context_valueB.setAutoDraw(True)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in valueBComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "valueB"-------
    for thisComponent in valueBComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    trials.addData('question_valueB.started', question_valueB.tStartRefresh)
    trials.addData('question_valueB.stopped', question_valueB.tStopRefresh)
    trials.addData('slider_valueB.response', slider_valueB.getRating())
    trials.addData('slider_valueB.rt', slider_valueB.getRT())
    trials.addData('slider_valueB.started', slider_valueB.tStartRefresh)
    trials.addData('slider_valueB.stopped', slider_valueB.tStopRefresh)
    trials.addData('stimB_2.started', stimB_2.tStartRefresh)
    trials.addData('stimB_2.stopped', stimB_2.tStopRefresh)
    win.mouseVisible = False
    trials.addData('Context_valueB.started', Context_valueB.tStartRefresh)
    trials.addData('Context_valueB.stopped', Context_valueB.tStopRefresh)
    # the Routine "valueB" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "percentage_trials"-------
    continueRoutine = True
    routineTimer.add(3.000000)
    # update component parameters for each repeat
    percentage = 'Has completado el '+str((trial_counter/80)*100)+'% del experimento!'
    info_percentage.setText(percentage)
    # keep track of which components have finished
    percentage_trialsComponents = [info_percentage]
    for thisComponent in percentage_trialsComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    percentage_trialsClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "percentage_trials"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = percentage_trialsClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=percentage_trialsClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *info_percentage* updates
        if info_percentage.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            info_percentage.frameNStart = frameN  # exact frame index
            info_percentage.tStart = t  # local t and not account for scr refresh
            info_percentage.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(info_percentage, 'tStartRefresh')  # time at next scr refresh
            info_percentage.setAutoDraw(True)
        if info_percentage.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > info_percentage.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                info_percentage.tStop = t  # not accounting for scr refresh
                info_percentage.frameNStop = frameN  # exact frame index
                win.timeOnFlip(info_percentage, 'tStopRefresh')  # time at next scr refresh
                info_percentage.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in percentage_trialsComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "percentage_trials"-------
    for thisComponent in percentage_trialsComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    trials.addData('info_percentage.started', info_percentage.tStartRefresh)
    trials.addData('info_percentage.stopped', info_percentage.tStopRefresh)
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
