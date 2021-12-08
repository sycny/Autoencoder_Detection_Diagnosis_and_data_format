def PMU_extract_nolimit(path):
    import sys
    import numpy as np
    import matplotlib.pyplot as plt
    from pandas import read_csv
    from keras.models import load_model
    from scipy.fftpack import fft
    from scipy.io import loadmat
    # %%
    f_s, Nn, num_time_periods, dataWindow = 2000, 200, 20, 1000
    slipSize = int(round(dataWindow - Nn) / num_time_periods)

    def feature_extract(testDataSetNI, f_s, Nn, num_time_periods, dataWindow, slipSize):
        dataFFT = np.zeros((18, num_time_periods))
        df = f_s / Nn  # resolution
        for num_i in range(0, num_time_periods):
            for signalNum in range(0, 6):
                dataFeature = testDataSetNI[signalNum, num_i * slipSize:num_i * slipSize + Nn]
                y = dataFeature - np.mean(dataFeature)
                han = np.hanning(Nn)  # Nn is the le
                y = han * y  # add a hanning window to prevent FFT leakage
                # ======================================================================
                f_x = np.arange(Nn) * df
                t_x0 = f_x - f_s / 2;
                fft_y = fft(y)  # FFT
                y_x0 = np.abs(fft_y) / len(y) * 2  # reshape the y-axis and obtain the real amplitude
                t_x = t_x0[int(len(t_x0) / 2):int(len(t_x0))]  # reshape the x-axis
                y_x = y_x0[0:int(len(t_x0) / 2)]  # for current
                # y_x_reshape=y_x0[int(len(t_x0)/2):int(len(t_x0)] # for voltage
                # y_x=y_x_reshape[::-1]

                # correct the amplitude and frequency 1
                for ii in range(0, len(t_x)):
                    if t_x[ii] <= 10:
                        y_x[ii] = 0.000001
                Ah = np.zeros((len(y_x), 1))
                Ah[:, 0] = y_x
                Amaxh = np.max(Ah)
                kh = np.argmax(Ah)
                if (kh >= 2) & (kh + 1 < int(Nn / 2)):
                    if Ah[kh - 1] > Ah[kh + 1]:
                        deltfh = (Ah[kh] / Ah[kh - 1] - 2) / (1 + Ah[kh] / Ah[kh - 1])
                        f0h = (kh + deltfh) * f_s / Nn
                        amh = 2 / np.sinc(deltfh) * Amaxh * (1 - deltfh ** 2)
                    else:
                        deltfh = (Ah[kh] / Ah[kh + 1] - 2) / (1 + Ah[kh] / Ah[kh + 1])
                        f0h = (kh - deltfh) * f_s / Nn
                        amh = 2 / np.sinc(deltfh) * Amaxh * (1 - deltfh ** 2)
                    y_x[kh] = amh
                    t_x[kh] = f0h
                    Ah[kh] = amh
                # ======================================================================
                positionF = np.argmax(y_x)
                # obtain THD
                for j in range(0, int(len(t_x))):
                    if abs(t_x[j] - 60 * 2) < df:  # df=f_s/Nn is resolution
                        J2 = j
                    if abs(t_x[j] - 60 * 3) < df:
                        J3 = j
                    if abs(t_x[j] - 60 * 4) < df:
                        J4 = j
                    if abs(t_x[j] - 60 * 5) < df:
                        J5 = j
                    if abs(t_x[j] - 60 * 6) < df:
                        J6 = j
                    if abs(t_x[j] - 60 * 7) < df:
                        J7 = j
                sumF = y_x[J2] ** 2 + y_x[J3] ** 2 + y_x[J4] ** 2 + y_x[J5] ** 2 + y_x[J6] ** 2 + y_x[J7] ** 2
                thd = np.sqrt(sumF) / y_x[positionF]
                dataFFT[signalNum * 3, num_i] = y_x[positionF]
                dataFFT[signalNum * 3 + 1, num_i] = t_x[positionF]
                dataFFT[signalNum * 3 + 2, num_i] = thd
                # I1_M,I1_F,I1_T,I2_M,I2_F,I2_T,I3_M,I3_F,I3_T,U1_M,U1_F,U1_T,
        return dataFFT

    # %% mean current vector (MCV)
    def mcv_def(testDataSetNI, f_s, Nn, num_time_periods, dataWindow, slipSize):
        cycle = round(f_s / 60)
        BarPmcv = np.zeros((1, num_time_periods))
        for num_i in range(0, num_time_periods):
            dataFeature = testDataSetNI[:, num_i * slipSize:num_i * slipSize + Nn]
            I_alpha = dataFeature[0, :] * np.sqrt(2 / 3) - np.sqrt(1 / 6) * dataFeature[1, :] - np.sqrt(
                1 / 6) * dataFeature[2, :]
            I_beta = dataFeature[1, :] * np.sqrt(1 / 2) - np.sqrt(1 / 2) * dataFeature[2, :]
            # --------------------------------------------------------------------------
            sNum = np.zeros((8, 1))
            for jj in range(cycle, Nn + 1):
                mcv_1 = np.sum(I_alpha[jj - cycle:jj]) / cycle
                mcv_2 = np.sum(I_beta[jj - cycle:jj]) / cycle
                if mcv_1 ** 2 + mcv_2 ** 2 >= 5:
                    if (mcv_1 >= 0) & (mcv_2 >= 0):  # 1st quadrant
                        if mcv_1 >= mcv_2:
                            sNum[0] = sNum[0] + 1
                        else:
                            sNum[1] = sNum[1] + 1
                    elif (mcv_1 < 0) & (mcv_2 >= 0):  # 2ed quadrant
                        if abs(mcv_1) <= mcv_2:
                            sNum[2] = sNum[2] + 1
                        else:
                            sNum[3] = sNum[3] + 1
                    elif (mcv_1 <= 0) & (mcv_2 < 0):  # 3rd quadrant
                        if abs(mcv_1) >= abs(mcv_2):
                            sNum[4] = sNum[4] + 1
                        else:
                            sNum[5] = sNum[5] + 1
                    else:  # 4th quadrant
                        if abs(mcv_1) <= abs(mcv_2):
                            sNum[6] = sNum[6] + 1
                        else:
                            sNum[7] = sNum[7] + 1
            # -------------------------------------------------------------------------
            S_f = np.zeros((8, 1))
            S_d = np.zeros((4, 1))
            if (np.sum(sNum) > 0) & ((np.sum(sNum) > (Nn - cycle) * 0.4)):
                S_f[0] = sNum[0] + sNum[1]  # the first region definition according to the paper
                S_f[1] = sNum[1] + sNum[2]
                S_f[2] = sNum[2] + sNum[3]
                S_f[3] = sNum[3] + sNum[4]
                S_f[4] = sNum[4] + sNum[5]
                S_f[5] = sNum[5] + sNum[6]
                S_f[6] = sNum[6] + sNum[7]
                S_f[7] = sNum[7] + sNum[0]
                feature_1 = np.max(S_f) / np.sum(sNum)
                S_d[0] = sNum[0] + sNum[4]  # the second region definition according to the paper
                S_d[1] = sNum[1] + sNum[5]
                S_d[2] = sNum[2] + sNum[6]
                S_d[3] = sNum[3] + sNum[7]
                feature_2 = np.max(S_d) / np.sum(sNum)
                BarPmcv[0, num_i] = np.max([feature_1, feature_2])
            else:
                BarPmcv[0, num_i] = 0
        return BarPmcv

    # %%
    def feature_collect(dataFFT, BarPmcv, num_time_periods):
        Rm1 = (dataFFT[0, :] - dataFFT[3, :]) ** 2 + (dataFFT[0, :] - dataFFT[6, :]) ** 2 + (
                dataFFT[3, :] - dataFFT[6, :]) ** 2
        Rm2 = (dataFFT[9, :] - dataFFT[12, :]) ** 2 + (dataFFT[9, :] - dataFFT[15, :]) ** 2 + (
                dataFFT[12, :] - dataFFT[15, :]) ** 2
        hatRm = (np.sqrt(Rm1) + np.sqrt(Rm2)) / 2
        BarRm1 = (np.log(hatRm + np.exp(1)) - 1) / 8
        BarRm2 = hatRm / 1000
        BarRm1 = BarRm1.reshape(1, BarRm1.shape[0])
        BarRm2 = BarRm2.reshape(1, BarRm2.shape[0])

        BarRf = np.abs(dataFFT[[1, 4, 7, 10, 13, 16], :] - 60) / 0.5  # 6 X 1

        BarT = dataFFT[[2, 5, 8, 11, 14, 17], :] / 0.05  # 6 X 1

        # ==============================================================================
        testX = np.zeros([15, num_time_periods])
        testX[0, :] = BarRm1
        testX[1, :] = BarRm2
        testX[2, :] = BarPmcv
        testX[3:9, :] = BarRf
        testX[9:15, :] = BarT
        return testX

    # %%
    TrainData = loadmat(path)
    Traindata = TrainData['measurement']
    samplepoint = np.linspace(50000, 800000, num=75000, endpoint=False, dtype=int)
    f_s, Nn, num_time_periods, dataWindow = 2000, 200, 20, 1000  # parameter definition
    slipSize = int(round(dataWindow - Nn) / num_time_periods)
    length_data = int(75000 / dataWindow)  # parameter definition
    #featuredata = np.zeros((7, 15, num_time_periods * length_data))
    featuredata = np.zeros((105, num_time_periods * length_data))
    for i in range(0, 7):
        dataCollect1 = Traindata[18 + i * 11:21 + i * 11, samplepoint]
        dataCollect2 = Traindata[21 + i * 11:24 + i * 11, samplepoint]
        dataCollect = np.vstack((dataCollect2,dataCollect1))  # In this code, the first three is current, the last three is voltage, this matters at MCV function
        featuredata2 = np.zeros((15, num_time_periods * length_data))
        print(i)
        for startNum in range(0, length_data):
            #print(startNum)
            testDataSetNI = dataCollect[0:6, startNum * dataWindow:startNum * dataWindow + dataWindow]
            dataFFT = feature_extract(testDataSetNI, f_s, Nn, num_time_periods, dataWindow, slipSize)
            BarPmcv = mcv_def(testDataSetNI, f_s, Nn, num_time_periods, dataWindow, slipSize)
            testX = feature_collect(dataFFT, BarPmcv, num_time_periods)
            featuredata2[:, num_time_periods * startNum:num_time_periods * startNum + num_time_periods] = testX
           #featuredata[i, :, :] = featuredata2
            featuredata[15 * i:15 * i + 15, :] = featuredata2
    '''
    #the following code is for PCC node
    # ------------------------------------------------------------------------------
    path1 = '/Users/ycs/Desktop/PhD first year/Fall2021 Task 1/lulu code and data/PVfarm data transformeringrid/newmeasurement_fault'
    path3 = '.mat'
    path2 = 1
    path = '%s%d%s'%(path1,path2,path3)
    TrainData = loadmat(path)
    Traindata = TrainData['measurement']
    samplepoint = np.linspace(50000,800000,num=75000,endpoint=False,dtype=int)
    dataCollect = Traindata[:,samplepoint] #chanege the row based on the mat file, it is supposed to be current and voltage
    f_s, Nn, num_time_periods, dataWindow = 2000, 200, 20, 1000  # parameter definition
    slipSize = int(round(dataWindow - Nn) / num_time_periods)  # parameter definition
    length_data = int(dataCollect.shape[1] / dataWindow)
    featuredata = np.zeros((15,num_time_periods*length_data))
    for startNum in range(0, length_data):
        print(startNum)
        testDataSetNI = dataCollect[0:6, startNum * dataWindow:startNum * dataWindow + dataWindow]
        dataFFT = feature_extract(testDataSetNI, f_s, Nn, num_time_periods, dataWindow, slipSize)
        BarPmcv = mcv_def(testDataSetNI, f_s, Nn, num_time_periods, dataWindow, slipSize)
        testX = feature_collect(dataFFT, BarPmcv, num_time_periods)
        featuredata[:,num_time_periods*startNum:num_time_periods*startNum+num_time_periods] = testX
    '''
    return featuredata
