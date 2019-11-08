#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


#mendefinisikan fungsi
#fungsi menggunakan sigmoid, artinya garisnya akan terbentuk antara 0 s.d. 1
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# In[3]:


#data train
training_inputs = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])
print(training_inputs)


# In[4]:


#data test
training_outputs = np.array([[0,1,1,0]]).T
print(training_outputs)


# In[5]:


#memakai fungsi random seed, artinya nilai random yang dihasilkan oleh saya dan ditempat lain sama
np.random.seed(1)
#hasil sesuai

#bikin random.random()
#artinya random diantara angka 0 s.d. 1
#lalu random.random((3,1))
#artinya random dalam bentuk 3 baris 1 kolom
#uji random.random
#np = np.random.random((3,2))
#hasil sesuai
#print(np)

#bikin variable
#isinya fungsi initialize weighting
synaptics_weights = 2 * np.random.random((3,1)) - 1
print(synaptics_weights)


# In[6]:


#np dot
#a = [[1,1],[1,0]]
#b = [[4,1],[2,2]]
#np.dot(a, b)
#kalikan angka 1 dengan 4,1 dan angka 1 dengan 2,2  lalu dijumlahkan dalam bentuk matriks
#hasilnya array([[6, 3], [4, 1]])

#hitung inputan sigmoid
#artinya mengalikan training inputs dengan synaptics weights berdasarkan fungsi np.dot
#muncul variable inputan sigmoid
#for iteration in range(1):
    #input_layer = training_inputs
    #input_sigmoid = np.dot(input_layer, synaptics_weights)
#print('Input Sigmoid: ')
#print(input_sigmoid)

#memasukkan variable inputan sigmoid ke dalam fungsi sigmoid
#1 / (1 + np.exp(-x))
#outputs = sigmoid(input_sigmoid)
#print('Output: ')
#print(outputs)


# In[7]:


#edit dengan iterasi 20000
#error disesuaikan dengan metode backpropagation
def sigmoid_derivative(x):
    return x * (1 - x) #sementara belum tau kenapa hasil turunan dari sigmoidnya hasilnya begitu

for iteration in range(20000):
    input_layer = training_inputs
    input_sigmoid = np.dot(input_layer, synaptics_weights)
    outputs = sigmoid(input_sigmoid)
    error = training_outputs - outputs
    adjustments = error * sigmoid_derivative(outputs)
    synaptics_weights += np.dot(input_layer.T, adjustments)
print('Synaptics weight after training: ')
print(synaptics_weights)
print('Outputs : ')
print(outputs)


# In[ ]:




