#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from matplotlib import pyplot as plt
import numpy as np


# In[21]:


#data terdiri dari lenght, width, type (0, 1)
data = [[3,   1.5, 1],
        [2,   1,   0],
        [4,   1.5, 1],
        [3,   1,   0],
        [3.5, 0.5, 1],
        [2,   0.5, 0],
        [5.5, 1,   1],
        [1,   1,   0]]

mystery_flower = [1, 1]


# In[4]:


#data[1] #cek row kedua 
#data[0] #cek row pertama
data [0] [1] #cek row pertama, kolom kedua


# In[5]:


#flower terdiri dari length, width, type
#length = w1
#width = w2
#bias = b
#ditentukan nilainya random
w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn()

#print
print('Nilai b: ')
print(b)


# In[6]:


#define fungsi dari sigmoid
def sigmoid(x):
    return 1/(1 + np.exp(-x))

#define derivative dari sigmoid(x)
def sigmoid_p(x):
    return sigmoid(x) * (1 - sigmoid(x)) 


# In[7]:


#bentuk angka dari -6 s.d 6, sebanyak 100 deret dengan interval sama
deret = np.linspace(-6, 6, 100)
Y = sigmoid_p(deret)
Y


# In[8]:


#bikin plot
plt.plot(deret, sigmoid(deret), c='r') #sigmoid sebelum derivative
fig = plt.plot(deret, Y, c='b') #sigmoid derivative #keterangannya hilang karena dijadikan variabel fig


# In[9]:


ri = np.random.randint(len(data))
ri
#len(data)
#data[ri]
point = data[ri]
point[0]


# In[12]:


#training loop
#for i in range (1, 1000):
    #print i
#fungsi loop sederhana

def train():
    #random init of weight
    w1 = np.random.randn() #length
    w2 = np.random.randn() #width
    b = np.random.randn() #bias
    
    iterations = 10000 #10ribu kali latihannya
    learning_rate = 0.1
    costs = [] #bentuk array karena []
    
    for i in range (iterations):
        #dapatkan random point
        #memilih dari kombinasi angka di data secara random
        #1 membentuk variabel ri = random data di jumlah data sepanjangan 1 s.d. len(data), data bilangan bulat
        ri = np.random.randint(len(data))
        #2 membentuk variabel point = mengambil data didalam variabel ri
        #bentuk data float sesuai dengan data
        point = data[ri]
        
        #membentuk variabel z = perhitungan feed forward perceptron
        #point[0] artinya kolom pertama dikalikan length
        z = point[0] * w1 + point[1] * w2 + b
        pred = sigmoid(z) #prediksi
        target = point[2] #targetnya 1 atau 0, red or blue
        
        #backpropagation
        cost = np.square(pred - target) #nilai error
        
        #print setiap 100 kali iterasi
        if i % 100 == 0:
            c = 0
            for j in range(len(data)):
                p = data[j]
                p_pred = sigmoid(w1 * p[0] + w2 * p[1] + b)
                c += np.square(p_pred - p[2])
            costs.append(c)
            
        dcost_dpred = 2 * (pred - target)
        dpred_dz = sigmoid_p(z)
        
        dz_dw1 = point[0]
        dz_dw2 = point[1]
        dz_db = 1
        
        dcost_dz = dcost_dpred + dpred_dz
        
        dcost_dw1 = dcost_dz * dz_dw1
        dcost_dw2 = dcost_dz * dz_dw2
        dcost_db = dcost_dz * dz_db
        
        w1 = w1 - learning_rate * dcost_dw1
        w2 = w2 - learning_rate * dcost_dw2
        b = b - learning_rate * dcost_db
    
    return costs, w1, w2, b


# In[19]:


costs, w1, w2, b = train()

fig = plt.plot(costs)

costs


# In[23]:


z = w1 * mystery_flower[0] + w2 * mystery_flower[1] + b
pred = sigmoid(z)

print(pred)
print('mendekati 0 adalah biru', 'mendekati 1 adalah merah')


# In[ ]:




