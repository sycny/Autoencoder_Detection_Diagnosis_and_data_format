#%%
import numpy as np
import pandas as pd
import tensorflow
from tensorflow import keras
from keras.layers import LayerNormalization
from pandas import read_csv
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from scipy.io import loadmat #load mat data on python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Conv1D, Conv1DTranspose
from tensorflow.keras.preprocessing import sequence
from PMU_extract import PMU_extract #need to add the 'from XXX' part or it will not be callable
#%%
def autoencoder_3(x_train):
    ## define the encoder
    encoder = Input(shape=(x_train.shape[1], x_train.shape[2]))
    e = Conv1D(filters=32, kernel_size=7, padding="same", strides=3, activation="relu")(encoder)
    e = Conv1D(filters=16, kernel_size=7, padding="same", strides=2, activation="relu")(e)
    e = Conv1D(filters=8, kernel_size=7, padding="same", strides=2, activation="relu")(e)
    ## bottleneck layer
    n_bottleneck = 3
    ## defining it with a name to extract it later
    bottleneck_layer = "bottleneck_layer"
    # can also be defined with an activation function, relu for instance
    bottleneck = Dense(n_bottleneck, name=bottleneck_layer)(e)

    ## define the decoder (in reverse)
    decoder = Conv1DTranspose(filters=8, kernel_size=7, padding="same", strides=2, activation="relu")(bottleneck)
    decoder = Conv1DTranspose(filters=16, kernel_size=7, padding="same", strides=2, activation="relu")(decoder)
    decoder = Conv1DTranspose(filters=32, kernel_size=7, padding="same", strides=3, activation="relu")(decoder)
    ## output layer
    output = Conv1DTranspose(filters=1, kernel_size=7, padding="same")(decoder)
    ## model
    model = Model(inputs=encoder, outputs=output)
    encoder = Model(inputs=model.input, outputs=bottleneck)
    model.summary()
    model.compile(loss="mse", optimizer="adam")
    history = model.fit(
        x_train,
        x_train,
        batch_size=4,
        epochs=20,
        verbose=1,
        validation_split=0.25
    )
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.show()
    return model
#%%
def autoencoder_2(x_train):
    model = keras.Sequential(
        [
            layers.Input(shape=(x_train.shape[1], x_train.shape[2])),
            layers.Conv1D(
                filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Dropout(rate=0.2),
            layers.Conv1D(
                filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Conv1DTranspose(
                filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Dropout(rate=0.2),
            layers.Conv1DTranspose(
                filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    model.summary()
    history = model.fit(
        x_train,
        x_train,
        epochs=50,
        batch_size=4,
        validation_split=0.2,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
        ],
    )
    #plt.plot(history.history["loss"], label="Training Loss")
    #plt.plot(history.history["val_loss"], label="Validation Loss")
    #plt.legend()
    #plt.show()
    return model

def standard(data):
    meansdata = np.mean(data)
    stddata = np.std(data)
    if stddata !=0:
        standdata = (data - meansdata) / stddata
    else:
        standdata = data
    #normaldata = normaldata.reshape((normaldata.shape[0], normaldata.shape[1], 1))
    return standdata

def normal(data):
    _range = np.max(data) - np.min(data)
    if _range !=0:
        normaldata = (data - np.min(data)) / _range
    else:
        normaldata = data
    return normaldata
#%%


def AEdetection(x_train,x_test,attacklen,i):
#%%
    model = autoencoder_2(x_train)
    #model = autoencoder_3(x_train)
    x_train_pred = model.predict(x_train)
    train_mae_loss_row = np.mean(np.abs(x_train_pred - x_train), axis=1)
    #train_mae_loss_column = np.mean(np.abs(x_train_pred - x_train), axis=0)
    # Get reconstruction loss threshold.
    threshold_row = np.mean(train_mae_loss_row)+1*np.std(train_mae_loss_row)
    #threshold_column = np.mean(train_mae_loss_column)+1*np.std(train_mae_loss_column)
    # find where are anomalies
    anomalies1 = train_mae_loss_row > threshold_row
    a = np.where(anomalies1)
    compare = np.intersect1d(np.where(anomalies1), np.arange(0,15))
    faultpercent=compare.shape[0]/np.sum(anomalies1)
    flag=True
    if faultpercent <= 1/7:
       flag=False
    #%%
    #find out when
    x_test_pred = model.predict(x_test)
    test_mae_loss_column = np.mean(np.abs(x_test_pred - x_test), axis=0)
    test_mae_loss_column = test_mae_loss_column.reshape((-1))

    #the attack happens during the 500-900 or 500-700, or 500-1500
    # Detect all the samples which are anomalies.
    anomalies2 = test_mae_loss_column > threshold_row# using row threshold is better
    #print("Column Number of anomaly samples: ", np.sum(anomalies2))
    #print("Column Indices of anomaly samples: ", np.where(anomalies2))
    normal = test_mae_loss_column <= threshold_row
    a = np.where(anomalies2)
    aa = np.where(normal)
    np.save(f"/Users/ycs/Desktop/PhD first year/Fall2021 Task 1/lulu code and data/PVfarm data/New PMU/AEresult/{i}_1116", a)
    b = np.arange(1700,attacklen)
    compare1 = np.intersect1d(np.where(anomalies2),b)
    TP = compare1.shape[0]
    c = np.arange(attacklen,2700)
    d = np.arange(0,1700)
    compare2a = np.intersect1d(np.where(anomalies2), c)
    compare2b = np.intersect1d(np.where(anomalies2), d)
    FP = compare2a.shape[0]+compare2b.shape[0]
    compare3a = np.intersect1d(np.where(normal), c)
    compare3b = np.intersect1d(np.where(normal), d)
    TN = compare3a.shape[0]+compare3b.shape[0]
    compare4 = np.intersect1d(np.where(normal),b)
    FN = compare4.shape[0]
    PPV = TP/(TP+FP)
    FNR = FN/(FN+TP)
    F1score = 2*TP/(2*TP+FP+FN)
    #print(np.sum(compare == b))
    detaccuracy = compare1.shape[0]/b.shape[0]
    #detrecall = b.shape[0]/np.sum(anomalies2)
    #detF1score =
    #%%
    return detaccuracy,PPV,FNR,F1score

Traindata = np.load( "/Users/ycs/Desktop/PhD first year/Fall2021 Task 1/lulu code and data/PVfarm data/PMUalldatanpz.npz" )
discardcase = [3, 7, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]
attacklens= 2100*np.ones(63,dtype='int')
attacklens[27]=1900
systemdown=[50,52,53,54,55,56,57,58,59]
attacklens[systemdown]=2700
AEresult = np.zeros((44,4))
j=0

for i in range(1,63):
    path = int(i)
    if i in discardcase:
        continue
    x_train = Traindata['a'][i,:,:]
    x_normal = x_train[0:105, 10:310]
    x_normal_long = np.tile(x_normal, (1, 4))
    x_train_long = np.hstack((x_normal_long, x_train))
    x_train_long = x_train_long.reshape((x_train_long.shape[0],x_train_long.shape[1],1))
    x_test_long = x_train_long[0:15, :]
    x_train_long = standard(x_train_long)
    x_train_long = normal(x_train_long)
    x_test_long = standard(x_test_long)
    x_test_long = normal(x_test_long)
    AEresult[j] =  AEdetection(x_train_long,x_test_long,attacklens[i],i)
    j = j+1

np.save("/Users/ycs/Desktop/PhD first year/Fall2021 Task 1/lulu code and data/PVfarm data/New PMU/AEresult1122", AEresult)
#np.save("/Users/ycs/Desktop/PhD first year/Fall2021 Task 1/lulu code and data/PVfarm data/New PMU/Where", where)