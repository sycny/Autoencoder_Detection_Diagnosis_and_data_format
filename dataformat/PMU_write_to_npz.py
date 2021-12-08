import sys
import numpy as np
import pandas as pd
import time
from scipy.io import loadmat
from pandas import read_csv

path1 = '/Users/ycs/Desktop/PhD first year/Fall2021 Task 1/lulu code and data/PVfarm data/New PMU/Nolimit/train_Xattack'
path2 = '.csv'
fault = [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,3,3,3,5,3,3,3,3,3,3,3,3,3,3,3,3,3,2,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,5,5]
IL = [] # The IL parameters
UC = [] # The UC parameters
fault = np.array(fault)
datasave = np.zeros((1,105,1500))# which contains 20s data
tagsave = np.zeros((1,4))
test = [2,3,10,11,31,46,48,49]
def tags(i):
    attacharray=np.zeros((15,4))
    #attacharray[:, 0] = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])# from the first to the last, the order is BarRm1, BarRm2, BarPmcv, BarRf(6), BarT(6)
    attacharray[:, 0] = time.time()
    attacharray[:, 1] = fault[i+2]
    attacharray[:, 2] = time.time()+15
    attacharray[:, 3] = time.time()+25
    #IL = []  # The IL parameters
    #UC = []  # The UC parameters
    return attacharray
#path3 = '.csv'


for i in range(-1,63):
    pathi = i
    if i in test:
        continue
    path = '%s%d%s' % (path1, pathi, path2)
    #pathfile = '%s%d%s' % (path0, pathi, path3)
    TrainData = read_csv(path)
    Traindata = TrainData.values
    #samplepoint = np.linspace(50000, 800000, num=75000, endpoint=False, dtype=int)
    dataCollect = Traindata[0:105]
    tagattach = tags(i)
    dataCollect = dataCollect.reshape(1, 105, 1500)
    datasave = np.vstack((datasave, dataCollect))
    tagsave = np.vstack((tagsave, tagattach))


datasave = datasave[1:,:,:]
tagsave = tagsave[1:,:]
np.savez("/Users/ycs/Desktop/PhD first year/Fall2021 Task 1/lulu code and data/PVfarm data/PMUtraindatanpz", a = datasave,b = tagsave)
print(datasave.shape)


datasave = np.zeros((1,105,1500))# which contains 20s data
tagsave = np.zeros((1,4))
for i in test:
    pathi = i
    path = '%s%d%s' % (path1, pathi, path2)
    #pathfile = '%s%d%s' % (path0, pathi, path3)
    TrainData = read_csv(path)
    Traindata = TrainData.values
    #samplepoint = np.linspace(50000, 800000, num=75000, endpoint=False, dtype=int)
    dataCollect = Traindata[0:105]
    tagattach = tags(i)
    dataCollect = dataCollect.reshape(1, 105, 1500)
    datasave = np.vstack((datasave, dataCollect))
    tagsave = np.vstack((tagsave, tagattach))



datasave = datasave[1:,:,:]
tagsave = tagsave[1:,:]
np.savez("/Users/ycs/Desktop/PhD first year/Fall2021 Task 1/lulu code and data/PVfarm data/PMUtestdatanpz", a = datasave,b = tagsave)
print(datasave.shape)


datasave = np.zeros((1,105,1500))# which contains 20s data
tagsave = np.zeros((1,4))
for i in range(1,63):
    pathi = i
    path = '%s%d%s' % (path1, pathi, path2)
    #pathfile = '%s%d%s' % (path0, pathi, path3)
    TrainData = read_csv(path)
    Traindata = TrainData.values
    #samplepoint = np.linspace(50000, 800000, num=75000, endpoint=False, dtype=int)
    dataCollect = Traindata[0:105]
    tagattach = tags(i)
    dataCollect = dataCollect.reshape(1, 105, 1500)
    datasave = np.vstack((datasave, dataCollect))
    tagsave = np.vstack((tagsave, tagattach))


datasave = datasave[1:,:,:]
tagsave = tagsave[1:,:]
np.savez("/Users/ycs/Desktop/PhD first year/Fall2021 Task 1/lulu code and data/PVfarm data/PMUalldatanpz", a = datasave,b = tagsave)
print(datasave.shape)
