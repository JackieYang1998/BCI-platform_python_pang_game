#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Kang Pan, Zhuokun Yang
# doing online inference without actually using online data

""" Motor imagery"""
import sys
import serial


from keras.models import load_model
from mne.io import read_raw_edf
import expyriment
import pywt
import math
from random import random
import numpy as np
import scipy.io
import yaml
import pickle
from scipy.fftpack import fft




f = open("./test_info.yaml",encoding ="utf-8")
conf = yaml.load(f,Loader = yaml.SafeLoader)



# parameters setting
n_blocks = 1                                 # the number of blocks
n_trials = conf["online_n_trials"]           # the number of trials
trial_interval = 2                           # the time interval between trials
radius = 20                                  # the radius of the cursor
velocity = 8                         
size   = [60,20]                             # the size of the rectangle

picks = conf["ChannelPicks"]                 # the channel need to process
winSize = conf["WinSize"]                    # the window size of a sample
high_pass = conf["high_pass"]                # high frequency in high pass filter
low_pass = conf["low_pass"]                  # low frequency in low pass filter
max_step = 100                               # stop condition 1: max moving steps
target_range = 40                            # stop condition 3: success
nfft = 256

# path
Subject = conf["Subject"]                    # subject id
OnlineTestDate = conf["OnlineTestDate"]      # date of the online test 
OnlineTestID = conf["OnlineTestID"]          # online test id
TestDate = conf["TestDate"]                  # the date when train data was recorded
TestMode = conf["TestMode"]                  # 1D task or 2D task
edf_path = "EEGdata/{}_online/{}/{}_{}_{}.edf".format(Subject,OnlineTestDate,Subject,OnlineTestDate,OnlineTestID)      # the path of edf file
root_path = "EEGdata/{}/{}/".format(Subject,TestDate)            # the path of root
model_path = root_path + "model.h5"                              # the path of model
save_dir = "EEGdata/{}_online/{}/{}/".format(Subject,TestDate,OnlineTestID)
save_name = "{}_online_task_{}.mat".format(Subject,TestMode)
save_path = save_dir + save_name                # the path where the events file will be saved      



if TestMode != '1D':
    print("Only 1D test is available")
    sys.exit()



class MotorImagery:
    def __init__(self, n_blocks, n_trials, trial_interval, velocity, radius, size, edf_path, model_path, save_path, picks, winSize, high_pass=0.1, low_pass=50):
        """

        :param n_blocks: the number of blocks
        :param n_trials: the number of trials
        :param trial_interval: the time interval between trials
        :param velocity: the velocity of the moving cursor
        :param radius: the radius of the cursor
        :param edf_path: the path of edf file
        :param model_path: the path of the lstm model
        """
        self.n_blocks = n_blocks
        self.n_trials = n_trials
        self.single_trials = int(n_trials / 4)
        self.trial_interval = trial_interval
        self.radius = radius
        self.size = size
        self.edf_path = edf_path
        self.model_path = model_path
        self.save_path = save_path
        self.exp = None
        self.BUAA_info = None
        self.text_init = None
        self.text_count_down = None
        self.text_trial_info = None
        self.dot = None
        self.rect = None
        self.velocity  = velocity
        self.readylen = []
        self.startlen = []
        self.endlen = []
        self.labels =[]
        self.state = []
        self.picks = picks
        self.winSize = winSize
        self.high_pass = high_pass
        self.low_pass = low_pass

    def initialize(self):
        self.exp = expyriment.design.Experiment(name="{} Pang Game: Cursor moving".format(TestMode))
        expyriment.control.initialize(self.exp)
        exp = self.exp
        self.dist = int((exp.screen.size[1] // 3)*0.75)
        self.border_size = [20,2*self.dist]
        self.border_size_u = [2*self.dist+40,20]



    def add_stimulation(self):
        dist = self.dist
        radius = self.radius
        n_blocks = self.n_blocks
        n_trials = self.n_trials
        trial_interval = self.trial_interval

        BUAA_info = expyriment.stimuli.TextLine(
            text="北航脑机接口实验平台", text_font="C:/Windows/Fonts/msyh.ttc", position=(-400, 320), text_size=40)
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
                text="Trial {}/{}".format(idx + 1, n_trials),
                text_font="C:/Windows/Fonts/msyh.ttc", position=(0, 0), text_size=100)
            )
            text_trial_info[-1].preload()
       
        dot  = expyriment.stimuli.Circle(radius=radius, colour=expyriment.misc.constants.C_GREEN, position=(0, 0 + dist))
        rect = expyriment.stimuli.Rectangle(size=size, colour=expyriment.misc.constants.C_GREY, position=(0, 0 - dist))        # dot0.preload()
        dot .preload()
        rect.preload()
        borderl = expyriment.stimuli.Rectangle(size=self.border_size, position=[-dist-10,0],
                                                    colour=expyriment.misc.constants.C_BLUE)
        borderr = expyriment.stimuli.Rectangle(size=self.border_size, position=[dist+10,0],
                                                colour=expyriment.misc.constants.C_BLUE)
        borderu = expyriment.stimuli.Rectangle(size=self.border_size_u, position=[0,dist+10],
                                            colour=expyriment.misc.constants.C_BLUE)                                    
        borderl.preload()
        borderr.preload()  
        borderu.preload()
        
        self.BUAA_info = BUAA_info
        self.text_init = text_init
        self.text_count_down = text_count_down
        self.text_trial_info = text_trial_info
        self.dot  = dot
        self.rect = rect
        self.borderl = borderl
        self.borderr = borderr
        self.borderu = borderu      

    
    def wpd(self,X): 
        coeffs = pywt.WaveletPacket(X,'db4',mode='symmetric',maxlevel=5)
        return coeffs
             
    def feature_bands(self,x):
    
        Bands = np.empty((8,1,x.shape[0],30)) # 8 freq band coefficients are chosen from the range 4-32Hz
        
        
        for ii in range(x.shape[0]):
            pos = []
            C = self.wpd(x[ii,:]) 
            pos = np.append(pos,[node.path for node in C.get_level(5, 'natural')])
            
            for b in range(1,9):
                Bands[b-1,0,ii,:] = C[pos[b]].data
            
        return Bands

    def feature_extract(self,x):

            feature = np.empty((1,1,x.shape[0],30)) 
            for ii in range(x.shape[0]):
                for time in range(30):
                    b_data = x[ii,time*24:(time+2)*24] 
                
                    Y = fft(b_data, 512)
                    Y = np.abs(Y)
                    ps = Y**2 / 512
                
                    feature[0,0,ii,time] = np.sum(ps[44:68]) # 8-12Hz
        
            return feature
    def feature_scalling(self,X,mmin, mmax):
        return (X - mmin) / (mmax - mmin)

    def feature_norm(self,X,mmean,sstd):
        return (X - mmean) / sstd

    def execute(self):
        exp = self.exp
        dist = self.dist
        n_trials = self.n_trials
        trial_interval = self.trial_interval
        BUAA_info = self.BUAA_info
        text_init = self.text_init
        text_count_down = self.text_count_down
        text_trial_info = self.text_trial_info
        
        dot  = self.dot
        rect = self.rect
        borderl = self.borderl
        borderr = self.borderr
        borderu = self.borderu

        picks = self.picks
        winSize = self.winSize
        high_pass = self.high_pass
        low_pass = self.low_pass
        
        Csp = pickle.load(open(root_path+'csp.txt', 'rb'))
        ss = pickle.load(open(root_path+'ss.txt', 'rb'))
        parameter = scipy.io.loadmat(root_path+'parameter.mat')
        mmean = parameter["mmean"][0]
        sstd = parameter["sstd"][0]

        
        # ser = serial.Serial('COM5', 9600, bytesize=8, parity='N', stopbits=1, timeout=1)
        
        # ser.write('1'.encode())
        # ser.readlines()
        # ser.write('1'.encode())
        # ser.readlines()


        # ser.write('4'.encode())

        # ser.write('3'.encode())
        # ser.readlines()
      

        # ser.write('2'.encode())
        # ser.readlines()
       

        # exp.clock.wait(10000)

        # ser.write('1'.encode()) 
        # ser.readlines()

        expyriment.control.start()

        exp.screen.clear()
        exp.screen.clear()
        exp.screen.update()
        exp.screen.clear()
        exp.screen.update()
        text_init.present(clear=False, update=False)
        BUAA_info.present(clear=False, update=False)
        exp.clock.wait(2000)
        exp.screen.update()
        exp.screen.clear()
        cnt_1, cnt_2 = 0, 0

        # the loop of trials
        for cnt in range(n_trials):
            # ser.write('4'.encode())
            # print (ser.readline())

            text_trial_info[cnt].present(clear=False, update=False)
            BUAA_info.present(clear=False, update=False)
            exp.screen.update()

            # save ready length
            record = read_raw_edf(self.edf_path, preload=False)
            record =list(record[:, :])[0]
            readylen = record.shape[1]
            self.readylen.append(readylen)

    
            exp.clock.wait(3000)
            exp.screen.clear()
            # count down
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
            record = read_raw_edf(self.edf_path, preload=True)
           
            # core code
          
            record =list(record[:, :])[0]
            startlen = record.shape[1]
            self.startlen.append(startlen)

            
     

            dot.position = [0, 0 + dist]
            rect.position = [0, 0 - dist]
            velocity_x = (-1+2*random())*velocity*0.9
            velocity_y = -math.sqrt(velocity**2-velocity_x**2)*0.5
            movement_dot  = [velocity_x, velocity_y]*2

            step = 1
            while 1:
                record = read_raw_edf(self.edf_path, preload=True)
                # step 1 : filter (band filter, notch filter 50Hz)
                # record.notch_filter(np.arange(50, 200, 50), picks=picks, fir_design='firwin')
                record.load_data()
                record._data=record._data[:,-5*winSize:]
                record.filter(high_pass, low_pass, fir_design='firwin')
                data = record._data
                data = data[picks,-winSize::4]

                
                wpd_data = self.feature_bands(data)
                burd_data = self.feature_extract(data)
                

                feature_data = np.concatenate((wpd_data, burd_data), axis = 0)

                feature_data = self.feature_norm(feature_data, mmean, sstd)
              
                fea_N = np.shape(feature_data)[0]

                X_test  = ss.transform(np.concatenate(tuple(Csp[x].transform(feature_data[x,:,:,:]) for x  in range(fea_N)),axis=-1))
                X_test  = X_test.reshape((np.shape(X_test)[0],np.shape(X_test)[1],-1))


                nn = load_model(root_path+'model.h5')

                # step 4 : prediction
        
                y_pred = nn.predict(X_test)
                   # if y_pred[0][0]>y_pred[0][1]:
                #     movement=-y_pred[0][0]
                # else:
                #     movement=y_pred[0][1]
                # movement -= 0.5
                # movement *= 2
                if y_pred.max(axis=1)<0.8:
                     movement=0
                else:
                    movement = (y_pred == y_pred.max(axis=1)[:,None]).astype(int)
                    movement = np.argmax(movement,axis=1)
                    movement = movement*2-1
                print("YPred:{}".format(movement))
                
                # ser.write(str(5.5+movement*0.5).encode())
                print(str(5.5+movement*0.5).encode())
                # while 1:
                #     s = ser.readline()
                #     print(s)
                #     if s=='OVER\r\n'.encode('utf_8'):
                #         break
               
                movement_rect = [movement*(2*velocity), 0]
                
                BUAA_info.present(clear=False, update=False)

                borderl.present(clear=False, update=False)
                borderr.present(clear=False, update=False)
                borderu.present(clear=False, update=False)
                erase_rect = expyriment.stimuli.Rectangle(size=rect.surface_size, position=rect.position,
                                                        colour=exp.background_colour)
                erase_dot = expyriment.stimuli.Rectangle(size=dot.surface_size, position=dot.position,
                                                        colour=exp.background_colour)
                
                
                
                # print(movement_dot, movement_rect)
                rect.move(movement_rect)

                if rect.position[0]>dist-size[0]*0.5:
                    rect.position=[dist-size[0]*0.5, 0 - dist]
                elif rect.position[0]<-dist+size[0]*0.5:
                    rect.position=[-dist+size[0]*0.5, 0 - dist]
                    
                dot.move(movement_dot)
                
                rect.present(clear=False, update=False)
                dot.present(clear=False, update=True)
                erase_rect.present(clear=False, update=False)
                erase_dot.present(clear=False, update=False)
                
                if  dot.position[1] <= rect.position[1]+radius+0.5*size[1] and dot.position[1] >= rect.position[1]:
                    if abs(dot.position[0] - rect.position[0])<0.5*size[0]:
                        movement_dot[1]*=-1
                        dot.position[1] = rect.position[1]+radius+0.5*size[1]
                        state = 1   #Success
                        cnt_1 = cnt_1 + 1
               
                if  dot.position[1]>dist-radius:
                    movement_dot[1]*=-1
                    dot.position[1] = dist-radius

                if  dot.position[0]>dist-radius:
                    movement_dot[0]*=-1
                    dot.position[0] = dist-radius

                if  dot.position[0]<-dist+radius:
                    movement_dot[0]*=-1
                    dot.position[0] = -dist+radius

                if  dot.position[1] <= rect.position[1]:
                    
                    break
                step = step + 1
            state = 0
            self.state.append(state)
            self.labels.append(velocity_x)
            # save end length
            record = read_raw_edf(self.edf_path, preload=False)
            record =list(record[:, :])[0]
            endlen = record.shape[1]
            self.endlen.append(endlen)
            exp.screen.clear()
            exp.screen.update()
            exp.screen.clear()
            exp.screen.update()

        print("Success trials: {}\n".format(cnt_1))
        print("Failure trials: {}\n".format(cnt_2))
        # ser.write('2'.encode())
        # print (ser.readlines())
        # ser.close()
        # end experiment
        expyriment.control.end()

    def save_data(self):
        scipy.io.savemat(self.save_path, {'readylen': self.readylen, 'startlen': self.startlen, 'endlen': self.endlen,
                                          'state': self.state, 'labels': self.labels})


if __name__ == '__main__':
    MI = MotorImagery(n_blocks, n_trials, trial_interval, velocity, radius, size, edf_path, model_path, save_path, picks, winSize, high_pass, low_pass)
    MI.initialize()
    MI.add_stimulation()
    MI.execute()
    MI.save_data()
