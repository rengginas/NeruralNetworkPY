#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
import numpy as np


# In[2]:


training_input = np.array([[0,0],
                         [0,1],
                         [1,0],
                         [1,1]]) #data inputan

training_solution = np.array ([[0,1,1,0]]).T #target

training_solution[2]


# In[3]:


def vis_data():
    plt.grid()
    
    for i in range(len(training_input)):
        c = 'r'
        if training_solution[i] == 0:
            c = 'b'
        plt.scatter([training_input[i][0]], [training_input [i][1]], c=c)
    
vis_data()


# In[4]:


def sigmoid (x):
    return 1/(1+np.exp(-x)) #sigmoid aktivasi

def sigmoid_p(x):
    return sigmoid(x) * (1-sigmoid(x)) #turunan sigmoid aktivasi


# In[5]:


w13 = 0.5
w14 = 0.9
w23 = 0.4
w24 = 1
w35 = -1.2
w45 = -1.1
b3 = 0.8
b4 = -0.1
b5 = 0.3
#nilai weighting dan bias, bisa dirandom
#dianggap local, jadi nilai ini nanti dimasukkan ke dalam rumus sewaktu mendefinisikan fungsinya


# In[6]:


def e(nilai_target, nilai_keluaran):
    return nilai_target - nilai_keluaran

#error(0,1) ngecek jika nilai target 0, dan nilai keluaran 1
#error setelah aktivasi, nilai target dikurangi nilai keluaran


# In[7]:


def eg(nilai_keluaran, error):
    return nilai_keluaran*(1-nilai_keluaran)*error

#error_gradien(0.5097, -0.5097) ngecek nilai keluaran 0.5097 dengan error -0.5097
#gradien error, mengalikan turunan dari nilai keluaran dengan error


# In[8]:


def weight_correction(learning_rate, input_terakhir, eg):
    return learning_rate*input_terakhir*eg
#weight_correction(0.1,0.520,-0.1274) ngecek nilai weight correction jika learning rate 0.1, input terakhir 0.520, dan eg -0.1274


# In[9]:


def update_weight(berat_sebelum, correction):
    return berat_sebelum + correction
#update_weight(0.1, -0.5) ngecek nilai update weight jika berat sebelum 0.1, dan correction -0.5


# In[10]:


def train():
    #masukkan seluruh nilai train
    w13 = 0.5
    w14 = 0.9
    w23 = 0.4
    w24 = 1
    w35 = -1.2
    w45 = -1.1
    b3 = 0.8
    b4 = -0.1
    b5 = 0.3
    #iterasi
    iterations = 10000
    learning_rate = 0.1
    sum_of_squared_error = []
    
    for j in range (iterations):
        sum_of_error = 0;
        for i in range (len(training_input)):
            
            x1 = training_input[i][0]
            x2 = training_input[i][1]
            training_solution[i]
            
            y3 = (x1*w13) + (x2*w23) - b3
            y3 = sigmoid(y3)
            
            y4 = (x1*w14) + (x2*w24) - b4
            y4 = sigmoid(y4)
            
            y5 = (y3*w35) + (y4*w45) - b5
            y5 = sigmoid(y5)
            
            error = e(training_solution[i], y5)
            sum_of_error += np.square(error)
            
            error_gradien = eg(y5, error) #error saat y5
            delta_w35 = weight_correction(learning_rate, y3, error_gradien)
            delta_w45 = weight_correction(learning_rate, y4, error_gradien)
            delta_b5 = weight_correction(learning_rate, -1, error_gradien)
            
            error_gradien3 = eg(y3, error_gradien*w35) #error saat y3
            delta_13 = weight_correction(learning_rate, x1, error_gradien3)
            delta_23 = weight_correction(learning_rate, x2, error_gradien3)
            delta_b3 = weight_correction(learning_rate, -1, error_gradien3)
            
            error_gradien4 = eg(y4, error_gradien*w45) #error saat y4
            delta_14 = weight_correction(learning_rate, x1, error_gradien4)
            delta_24 = weight_correction(learning_rate, x2, error_gradien4)
            delta_b4 = weight_correction(learning_rate, -1, error_gradien4)
            
            w13 = update_weight(w13, delta_13)
            w14 = update_weight(w14, delta_14)
            w23 = update_weight(w23, delta_23)
            w24 = update_weight(w24, delta_24)
            w35 = update_weight(w35, delta_w35)
            w45 = update_weight(w45, delta_w45)
            b3 = update_weight(b3, delta_b3)
            b4 = update_weight(b4, delta_b4)
            b5 = update_weight(b5, delta_b5)
            
        if j % 100 == 0:
            sum_of_squared_error.append(sum_of_error)
    return sum_of_squared_error, w13, w14, w23, w24, w35, w45, b3, b4, b5


# In[11]:


sum_of_squared_error, w13, w14, w23, w24, w35, w45, b3, b4, b5 = train()
fig = plt.plot (sum_of_squared_error)
#bikin plot


# In[12]:


def prediction (x1,x2):
    y3 = (x1*w13) + (x2*w23) - b3
    y3 = sigmoid(y3)
    y4 = (x1*w14) + (x2*w24) - b4
    y4 = sigmoid(y4)
    y5 = (y3*w35) + (y4*w45) - b5
    y5 = sigmoid(y5)
    return y5


# In[13]:


prediction(1,1)


# In[20]:


for x in np.linspace(0,1,10):
    for y in np.linspace(0,1,10): #10 jumlah data yang diprediksi (len)
        pred = prediction(x,y)
        c = 'b'
        if pred > 0.5:
            c = 'r'
        plt.scatter([x],[y], c=c, alpha=0.8) #alpha untuk terang


# In[ ]:




