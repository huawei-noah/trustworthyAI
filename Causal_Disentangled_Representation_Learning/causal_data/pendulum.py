#Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#This program is free software; 
#you can redistribute it and/or modify
#it under the terms of the MIT License.
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.

import matplotlib.pyplot as plt
import os

import matplotlib.image as mpimg
import random
import math
import numpy as np
import pandas as pd 
if not os.path.exists('./causal_data/pendulum/'): 
  os.makedirs('./causal_data/pendulum/train/')
  os.makedirs('./causal_data/pendulum/test/')

def projection(theta, phi, x, y, base = -0.5):
    b = y-x*math.tan(phi)
    shade = (base - b)/math.tan(phi)
    return shade
# 
scale = np.array([[0,44],[100,40],[7,7.5],[10,10]])
count = 0
empty = pd.DataFrame(columns=['i', 'j', 'shade','mid'])
for i in range(-40,44):#pendulum
    for j in range(60,148):#light
        if j == 100:
            continue
        plt.rcParams['figure.figsize'] = (1.0, 1.0)
        theta = i*math.pi/200.0
        phi = j*math.pi/200.0
        x = 10 + 8*math.sin(theta)
        y = 10.5 - 8*math.cos(theta)

        ball = plt.Circle((x,y), 1.5, color = 'firebrick')
        gun = plt.Polygon(([10,10.5],[x,y]), color = 'black', linewidth = 3)

        light = projection(theta, phi, 10, 10.5, 20.5)
        sun = plt.Circle((light,20.5), 3, color = 'orange')


        #calculate the mid index of 
        ball_x = 10+9.5*math.sin(theta)
        ball_y = 10.5-9.5*math. cos(theta)
        mid = (projection(theta, phi, 10.0, 10.5)+projection(theta, phi, ball_x, ball_y))/2
        shade = max(3,abs(projection(theta, phi, 10.0, 10.5)-projection(theta, phi, ball_x, ball_y)))

        shadow = plt.Polygon(([mid - shade/2.0, -0.5],[mid + shade/2.0, -0.5]), color = 'black', linewidth = 3)
        
        ax = plt.gca()
        ax.add_artist(gun)
        ax.add_artist(ball)
        ax.add_artist(sun)
        ax.add_artist(shadow)
        ax.set_xlim((0, 20))
        ax.set_ylim((-1, 21))
        new=pd.DataFrame({
                  'i':(i-scale[0][0])/(scale[0][1]-0),
                  'j':(j-scale[1][0])/(scale[1][1]-0),
                  'shade':(shade-scale[2][0])/(scale[2][1]-0),
                  'mid':(mid-scale[2][0])/(scale[2][1]-0)
                  },
                  
                 index=[1])
        empty=empty.append(new,ignore_index=True)
        plt.axis('off')
        if count == 4:
          plt.savefig('./causal_data/pendulum/test/a_' + str(int(i)) + '_' + str(int(j)) + '_' + str(int(shade)) + '_' + str(int(mid)) +'.png',dpi=96)
          count = 0
        else:
          plt.savefig('./causal_data/pendulum/train/a_' + str(int(i)) + '_' + str(int(j)) + '_' + str(int(shade)) + '_' + str(int(mid)) +'.png',dpi=96)
        plt.clf()
        count += 1



