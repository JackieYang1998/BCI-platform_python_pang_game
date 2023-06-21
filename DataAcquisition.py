#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# author: Kang Pan, Zhuokun Yang

""" Motor imagery"""

from random import random, randint
from tkinter.tix import InputOnly
from cv2 import distanceTransform
from   mne.io import read_raw_edf
import expyriment
import numpy  as np
import scipy.io
import os
import yaml
import sys
import math


f    = open("./test_info.yaml",encoding = 'utf-8')
conf = yaml.load(f,Loader = yaml.SafeLoader)

# parameters setting
n_blocks       = 1                                 # the number of blocks
n_trials       = conf["n_trials"]                  # the number of trials
trial_interval = 3                            # the time interval between trials
velocity       = 0.7                                # the velocity of the moving cursor
radius         = 20                                # the radius of the cursor
size           = [60,20]

Subject   = conf["Subject" ]                       # subject id
TestDate  = conf["TestDate"]                       # date of the data recorded test       
TestID    = conf["TestID"  ]                       # test id
TestMode  = conf["TestMode"]                       # 1D task or 2D task

save_dir  = "EEGdata/{}/{}/{}/".format(Subject,TestDate,TestID)
edf_name  = "{}_{}_{}.edf".format(Subject,TestDate,TestID,Subject,TestDate,TestID)    # the path of edf file
evt_name  = "{}_task_{}.mat".format(Subject,TestMode)
save_path = save_dir + evt_name                     # the path where the events file will be saved      
edf_path  = save_dir + edf_name 

if TestMode != '1D':
    print("Only 1D test is available")
    sys.exit()



class MotorImagery:
    def __init__(self, n_blocks, n_trials, trial_interval, velocity, radius, size, edf_path, save_path):
        """

        :param n_blocks: the number of blocks
        :param n_trials: the number of trials
        :param trial_interval: the time interval between trials
        :param velocity: the velocity of the moving cursor
        :param radius: the radius of the cursor
        :param edf_path: the path of edf file
        :param save_path: the path of the date need to be saved
        """
        self.n_blocks = n_blocks
        self.n_trials = n_trials
        self.single_trials  = int(n_trials / 2)
        self.labels = np.array([0] * self.single_trials + [1] * self.single_trials)
        np.random.shuffle(self.labels)
        self.str_labels = ["左", "右"] 
        self.trial_interval = trial_interval
        self.velocity  = velocity
        self.radius    = radius
        self.size = size
        self.edf_path  = edf_path
        self.save_path = save_path
        self.exp = None
        self.BUAA_info = None
        self.text_init = None
        self.text_count_down = None
        self.text_trial_info = None
        self.dot0 = None
        self.dot1 = None
        self.dot2 = None
        self.dot3 = None
        self.dot  = None
        self.readylen = []
        self.startlen = []
        self.endlen   = []

    def initialize(self):
       
        expyriment.control.defaults.window_size=(600,600)
        self.exp = expyriment.design.Experiment(name="{} Pang Game: Cursor moving".format(TestMode))
        expyriment.control.initialize(self.exp)
        exp = self.exp
        self.dist = int((exp.screen.size[1] // 3)*0.75)

    def add_stimulation(self):
        radius   = self.radius
        size     = self.size
        n_blocks = self.n_blocks
        n_trials = self.n_trials
        labels   = self.labels
        trial_interval = self.trial_interval
        str_labels     = self.str_labels
        dist = self.dist
        BUAA_info = expyriment.stimuli.TextLine(
            text="北航脑机接口实验平台", text_font="C:/Windows/Fonts/msyh.ttc", position=(-400, 300), text_size=40)
        text_init = expyriment.stimuli.TextLine(
            text="{}个blocks，每个block包含{}个trials，时间间隔{}秒".format(n_blocks, n_trials, trial_interval),
            text_font="C:/Windows/Fonts/msyh.ttc", position=(0, 0), text_size=50)
        BUAA_info.preload()
        text_init.preload()
        text_count_down = []
        for idx in range(trial_interval):
            text_count_down.append(expyriment.stimuli.TextLine(
                text="{}".format(trial_interval - idx),
                text_font="C:/Windows/Fonts/msyh.ttc", position=(0, 0), text_size=200)
            )
            text_count_down[-1].preload()
     
        text_trial_info = []
        for idx in range(n_trials):
            text_trial_info.append(expyriment.stimuli.TextLine(
                text="Trial {}/{} ： {}".format(idx + 1, n_trials, str_labels[labels[idx]]),
                text_font="C:/Windows/Fonts/msyh.ttc", position=(0, 0), text_size=100)
            )
            text_trial_info[-1].preload()
        dot  = expyriment.stimuli.Circle(radius=radius, colour=expyriment.misc.constants.C_GREEN, position=(0, 0 + dist))
        rect = expyriment.stimuli.Rectangle(size=size, colour=expyriment.misc.constants.C_GREY, position=(0, 0 - dist))        # dot0.preload()
        dot .preload()
        rect.preload()
        self.BUAA_info = BUAA_info
        self.text_init = text_init
        self.text_count_down = text_count_down
        self.text_trial_info = text_trial_info
        self.dot  = dot
        self.rect = rect

    def execute(self):
        exp = self.exp
        dist = self.dist
        labels   = self.labels
        n_trials = self.n_trials
        trial_interval = self.trial_interval
        velocity  = self.velocity
        BUAA_info = self.BUAA_info
        text_init = self.text_init
        text_count_down = self.text_count_down
        text_trial_info = self.text_trial_info

        dot  = self.dot
        rect = self.rect

        # start experiment
        expyriment.control.start()

        exp.screen.clear()
        exp.screen.update()
        exp.screen.clear()
  
        exp.screen.update()
        text_init.present(clear=False, update=False)
        BUAA_info.present(clear=False, update=False)
        exp.screen.update()
        exp.clock.wait(1000)
        exp.screen.clear()
        # the loop of trials
        for cnt in range(n_trials):
            # os.system('PAUSE')``
            text_trial_info[cnt].present(clear=False, update=False)
            BUAA_info.present(clear=False, update=False)
            exp.screen.update()

            # save ready length
            record = read_raw_edf(self.edf_path, preload=False)
            record =list(record[:, :])[0]
            self.readylen.append(record.shape[1])

            exp.clock.wait(1000)
            exp.screen.clear()
            # count down display
            for idx in range(trial_interval):
                text_count_down[idx].present(clear=False, update=False)
                BUAA_info.present(clear=False, update=False)
                exp.screen.update()
                exp.clock.wait(1000)
                exp.screen.clear()
            exp.screen.clear()
            exp.screen.update()
            exp.screen.clear()
            exp.screen.update()
            # save start length
            record = read_raw_edf(self.edf_path, preload=False)
            record = list(record[:, :])[0]
            self.startlen.append(record.shape[1])
           
            dot.position = [0, 0 + dist]
            rect.position = [0, 0 - dist]

            velocity_x = 2*(labels[cnt]-0.5)*velocity
            velocity_y = -velocity
            movement_rect = [velocity_x, 0]
            movement_dot  = [velocity_x, velocity_y]
            while dot.position[1] > rect.position[1]+radius+0.5*size[1]:
                BUAA_info.present(clear=False, update=False)

                erase_rect = expyriment.stimuli.Rectangle(size=rect.surface_size, position=rect.position,
                                                        colour=exp.background_colour)
                erase_dot = expyriment.stimuli.Rectangle(size=dot.surface_size, position=dot.position,
                                                        colour=exp.background_colour)
                
                rect.move(movement_rect)
                dot.move(movement_dot)
              
                rect.present(clear=False, update=False)
                dot.present(clear=False, update=True)
                erase_rect.present(clear=False, update=False)
                erase_dot.present(clear=False, update=False)

           
            record = read_raw_edf(self.edf_path, preload=False)
            record =list(record[:, :])[0]
            self.endlen.append(record.shape[1])
            exp.screen.clear()
            exp.screen.update()
            exp.screen.clear()
            exp.screen.update()

        # end experiment
        expyriment.control.end()
        self.labels = labels
        print(self.readylen)
        print(self.labels)



    def save_data(self):
        if not(os.path.exists(save_dir)):
            os.makedirs(save_dir)
        if os.path.exists(self.save_path):
            os.remove(self.save_path)
        scipy.io.savemat(self.save_path, {'readylen': self.readylen, 'startlen': self.startlen,
                                         'endlen': self.endlen,  'labels': self.labels})


if __name__ == '__main__':
    MI = MotorImagery(n_blocks, n_trials, trial_interval, velocity, radius, size, edf_path, save_path)
    MI.initialize()
    MI.add_stimulation()
    MI.execute()
    MI.save_data()
