import sys
import numpy as np
import pandas as pd
import time
from scipy.io import loadmat


path1 = '/Users/ycs/Desktop/PhD first year/Fall2021 Task 1/lulu code and data/PVfarm data/grid_attack'
#path0 = '/Users/ycs/Desktop/PhD first year/Fall2021 Task 1/lulu code and data/PVfarm data/New PMU/Withlimit/train_Xattack'
path2 = '.mat'
fault = [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,3,3,3,5,3,3,3,3,3,3,3,3,3,3,3,3,3,2,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,5,5]
IL = [] # The IL parameters
UC = [] # The UC parameters
fault = np.array(fault)
datasave = np.zeros((1,6,800000))
tagsave = np.zeros((1,4))# which contains 20s data
test = [2,3,10,11,31,46,48,49]
def tags(i):
    attacharray=np.zeros((1,4))
    #attacharray[:, 0] = np.array([1,2,3,4,5,6])# from the first to the last, the order is V1,V2,V3,I1,I2,I3
    attacharray[:, 0] = time.time()
    attacharray[:, 1] = fault[i+2]
    attacharray[:, 2] = time.time()+15
    attacharray[:, 3] = time.time()+25
    #attacharray[:, 5] = IL[i]
    #attacharray[:, 6] = UC[i]
    return attacharray
#path3 = '.csv'


for i in range(-1,63):
    pathi = i
    if i in test:
        continue
    path = '%s%d%s' % (path1, pathi, path2)
    #pathfile = '%s%d%s' % (path0, pathi, path3)
    TrainData = loadmat(path)
    Traindata = TrainData['gridvoltage']
    #samplepoint = np.linspace(50000, 800000, num=75000, endpoint=False, dtype=int)
    dataCollect = Traindata[1:7]
    tagattach = tags(i)
    dataCollect = dataCollect.reshape(1,6,800000)
    datasave = np.vstack((datasave,dataCollect))
    tagsave = np.vstack((tagsave, tagattach))

datasave = datasave[1:,:,:]
tagsave = tagsave[1:,:]
np.savez("/Users/ycs/Desktop/PhD first year/Fall2021 Task 1/lulu code and data/PVfarm data/traindatanpz", a = datasave,b = tagsave)
print(datasave.shape)

datasave = np.zeros((1,6,800000))
tagsave = np.zeros((1,4))# which contains 20s data

for i in test:
    pathi = i
    path = '%s%d%s' % (path1, pathi, path2)
    #pathfile = '%s%d%s' % (path0, pathi, path3)
    TrainData = loadmat(path)
    Traindata = TrainData['gridvoltage']
    #samplepoint = np.linspace(50000, 800000, num=75000, endpoint=False, dtype=int)
    dataCollect = Traindata[1:7]
    tagattach = tags(i)
    dataCollect = dataCollect.reshape(1,6,800000)
    datasave = np.vstack((datasave,dataCollect))
    tagsave = np.vstack((tagsave, tagattach))

datasave = datasave[1:,:,:]
tagsave = tagsave[1:,:]
np.savez("/Users/ycs/Desktop/PhD first year/Fall2021 Task 1/lulu code and data/PVfarm data/testdatanpz", x = datasave,y = tagsave)
print(datasave.shape)
