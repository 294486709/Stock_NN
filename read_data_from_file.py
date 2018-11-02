import pandas as pd
from indicators2 import *
#import pandas_datareader.data as web
import datetime as dt
import numpy as np
import indicators as id
import copy
import csv
import os
import copy
import shutil
volumn_index = 5
num_index = 25
raw_data_length = 6
outputnumber = 4
prev = 17


def gety():
    pass


# MACD=1,KDJ=0,RSI=0,ADX=0,CCI=0,BBANS=0,AD=0,OBV=0,LB=1,CMF=0,FI=0,EOM=0,VPT=0,ATR=0,KC=0,DC=0,VI=0,TRIX=0,MI=0,DPO=0,KST=0,ICH=0,TSI=0,DR=0,CR=0
def be_called(Scode, data,margin,period,upvalue,val):
    MACD = 1
    KDJ = 1
    RSI = 1
    ADX = 1
    CCI = 1
    BBANS = 1
    AD = 1
    OBV = 1
    LB = 1
    CMF = 1
    FI = 1
    EOM = 1
    VPT = 1
    ATR = 1
    KC = 1
    DC = 1
    VI = 1
    TRIX = 1
    MI = 1
    DPO = 1
    KST = 1
    ICH = 1
    TSI = 1
    DR = 1
    CR = 1
    ress = []


    #############
    #MACD NUM=9 0-8
    #############
    if MACD == 1:
        data1 = data.rename(index=str,columns={'Close':'<CLOSE>'})
        #print(data1.head(30))
        data1 = id.macd(data1,26,12,9,'<ClOSE>')
        data_MACD = pandas_to_list(data1)
        for i in data_MACD:
            ress.append(i)

    #############
    #KDJ NUM=3 9-11
    #############
    if KDJ == 1:
        data1 = data1.rename(index=str,columns={'<CLOSE>':'Close'})
        #print(data1.head(30))
        for i in range(3):
            data_KDJ = zhibiao_kdj(data1,9,3,3)[i].values
            data_KDJ = list(data_KDJ)
            ress.append(data_KDJ)

    #############
    #RSI NUM=1 12
    #############

    data2 = copy.copy(data)
    if RSI == 1:
        data_rsi = rsi(data2).tolist()
        ress.append(data_rsi)

    #############
    #ADX NUM=3 13-15
    #############
    if ADX == 1:
        data3 = copy.copy(data)
        data_adx = []
        data_adx.append(adx(data3).tolist())
        data_adx.append(adx_pos(data3).tolist())
        data_adx.append(adx_neg(data3).tolist())
        for i in data_adx:
            ress.append(i)

    ############
    #CCI NUM=1 16
    ############
    if CCI == 1:
        data_cci = cci(data).tolist()
        ress.append(data_cci)

    ############
    #BBANS NUM=3 17-19
    ############
    if BBANS == 1:
        data_bbans = []
        data_bbans.append(bollinger_hband(data2).tolist())
        data_bbans.append(bollinger_lband(data2).tolist())
        data_bbans.append(bollinger_mavg(data2).tolist())
        for i in data_bbans:
            ress.append(i)

    ############
    #AD NUM=1 20
    ############
    if AD == 1:
        data_adi = acc_dist_index(data2).tolist()
        ress.append(data_adi)

    ############
    #OBV NUM=2 21 22
    ############
    if OBV == 1:
        data_obv = []
        data_obv.append(on_balance_volume(data2).tolist())
        data_obv.append(on_balance_volume_mean(data2).tolist())
        for i in data_obv:
            ress.append(i)

    #################################################################################3
    ############
    #CMF NUM=1 23
    ############
    if CMF == 1:
        data_cmf = chaikin_money_flow(data2).tolist()
        ress.append(data_cmf)

    ############
    #FI NUM=1 24
    ############
    if FI == 1:
        data_fi = force_index(data2).tolist()
        ress.append(data_fi)

    ############
    #EOM EMV NUM=1 25
    ############
    if EOM == 1:
        data_eomemv = ease_of_movement(data2).tolist()
        ress.append(data_eomemv)
    ############
    #VPT NUM=1 26
    ############
    if VPT == 1:
        data_vpt = volume_price_trend(data).tolist()
        ress.append(data_vpt)
    ############
    #ATR NUM=1 27
    ############
    if ATR == 1:
        data_atr = average_true_range(data2).tolist()
        ress.append(data_atr)
    ############
    #KC NUM=2 29
    ############
    if KC == 1:
        data_kc = []
        data_kc.append(keltner_channel_hband(data2).tolist())
        data_kc.append(keltner_channel_lband(data2).tolist())
        for i in data_kc:
            ress.append(i)
    '''
    ############
    #DC NUM=2 30 31
    ############
    '''
    if DC == 1:
        data_dc = []
        data_dc.append(donchian_channel_hband(data2).tolist())
        data_dc.append(donchian_channel_lband(data2).tolist())
        for i in data_dc:
            ress.append(i)
    '''
    ############
    #VI NUM=2 32 33
    ############
    '''
    if VI == 1:
        data_vi = []
        data_vi.append(vortex_indicator_neg(data2).tolist())
        data_vi.append(vortex_indicator_pos(data2).tolist())
        for i in data_vi:
            ress.append(i)
    '''
    ############
    #TRIX NUM=1 34
    ############
    '''
    if TRIX == 1:
        data_trix = trix(data2).tolist()
        ress.append(data_trix)
    '''
    ############
    #MI NUM=1 35
    ############
    '''
    if MI == 1:
        data_mi = mass_index(data2).tolist()
        ress.append(data_mi)
    '''
    ############
    #DPO NUM=1 36
    ############
    '''
    if DPO == 1:
        data_dpo = dpo(data2).tolist()
        ress.append(data_dpo)
    '''
    ############
    #KST NUM=2 37 38
    ############
    '''
    if KST == 1:
        data_kst = []
        data_kst.append(kst(data2).tolist())
        data_kst.append(kst_sig(data2).tolist())
        for i in data_kst:
            ress.append(i)
    '''
    ############
    #Ichimoku NUM=2 39 40
    ############
    '''
    if ICH == 1:
        data_ich = []
        data_ich.append(ichimoku_a(data2).tolist())
        data_ich.append(ichimoku_b(data2).tolist())
        for i in data_ich:
            ress.append(i)
    '''
    ############
    #MFI NUM=1 41
    ############
    '''
    if FI == 1:
        data_mfi = money_flow_index(data2).tolist()
        ress.append(data_mfi)
    '''
    ############
    #TSI NUM=1 42
    ############
    '''
    if TSI == 1:
        data_tsi = tsi(data2).tolist()
        ress.append(data_tsi)
    '''
    ############
    #DR NUM=1 43
    ############
    '''
    if DR == 1:
        data_dr = daily_return(data2).tolist()
        ress.append(data_dr)
    '''
    ############
    #CR NUM=1 44
    ############
    '''
    if CR == 1:
        data_cr = cumulative_return(data2).tolist()
        ress.append(data_cr)
    closelist = data['Volume'].tolist()
    if LB == 1:
        numexp = len(closelist) - 5
        vb = []
        for i in range(5):
            vb.append(np.nan)
        for i in range(5,5+numexp):
            try:
                cc = closelist[i]/(closelist[i-5]+closelist[i-4]+closelist[i-3]+closelist[i-2]+closelist[i-1])
            except ZeroDivisionError:
                cc = 0
            vb.append(cc)

        ress.append(vb)
    # remove five index and back up close for Y
    res = []

    ress.pop(2)
    ress.pop(2)
    ress.pop(0)
    for i in range(len(ress)):
        if i in val:
            res.append(copy.copy(ress[i]))
        else:
            pass
    res1 = copy.copy(res)
    res = copy.copy(res1)
    for i in range(len(ress[0])):
        try:
            ress[0][i] = (ress[1][i] - ress[0][i])/ress[0][i]
        except ZeroDivisionError:
            ress[0][i] = 0
        if ress[0][i] > 0.03:
            temp = np.zeros([5])
            temp[4] = 1
            ress[1][i] = temp
        elif ress[0][i] > 0.01 and ress[0][i]<= 0.03:
            temp = np.zeros([5])
            temp[3] = 1
            ress[1][i] = temp
        elif ress[0][i] > -0.01 and ress[0][i]<= 0.01:
            temp = np.zeros([5])
            temp[2] = 1
            ress[1][i] = temp
        elif ress[0][i] > -0.03 and ress[0][i]<= -0.01:
            temp = np.zeros([5])
            temp[1] = 1
            ress[1][i] = temp
        else:
            temp = np.zeros([5])
            temp[0] = 1
            ress[1][i] = temp
    value_rate = ress[0]
    plus3 = 0
    plus1 = 0
    noplus = 0
    minus1 = 0
    minus3 = 0
    value_falg = ress[1]
    for i in range(len(value_rate)):
        if value_rate[i]*100 > 3:
            plus3 += 1
        elif value_rate[i]*100 > 1 and value_rate[i]*100 < 3:
            plus1 += 1
        elif value_rate[i]*100 > -1 and value_rate[i]*100 < 1:
            noplus += 1
        elif value_rate[i]*100 > -3 and value_rate[i]*100 < -1:
            minus1 += 1
        else:
            minus3 += 1
    avg = len(value_rate)/5
    stand_dev = 0.2*pow(plus3-avg,2) + 0.2*pow(plus1-avg,2) + 0.2*pow(noplus-avg,2) + 0.2*pow(minus1-avg,2) + 0.2*pow(minus3-avg,2)
    ff = open('stand_dev.txt','a')
    ff.write(Scode)
    ff.write(',')
    ff.write(str(stand_dev))
    ff.write('\n')
    for i in range(len(res)):
        for j in range(60):
            res[i].pop(0)
    for i in range(60):
        value_falg.pop(0)
    sample = []
    ylabel = []
    for i in range(len(res[0]) - margin - period + 1):
        sampletemp = []
        for j in range(len(res)):
            for k in range(i,i+margin):
                sampletemp.append(res[j][k])
        sample.append(sampletemp)
        ylabel.append(value_falg[i+margin])
    oneday = []
    for i in range(len(res)):
        for j in range(len(res[0])-margin,len(res[0])):
            oneday.append(res[i][j])
    np_oneday = np.zeros([1,len(oneday)])
    for i in range(len(oneday)):
        np_oneday[0][i] = oneday[i]
    ylabel = np.array(ylabel)
    testingsamples_np = np.array(sample)


    os.mkdir(os.getcwd()+'/Stock_Data_IDV/'+Scode)
    np.save(os.getcwd()+'/Stock_Data_IDV/'+Scode+'/train.npy',testingsamples_np)
    np.save(os.getcwd()+'/Stock_Data_IDV/'+Scode+'/ylabel.npy',ylabel)
    np.save(os.getcwd()+'/Stock_Data_IDV/'+Scode+'/oneday.npy',np_oneday)

    return 0


def main(Scode,data,margin,period,upvalue,val):
    be_called(Scode,data,margin,period,upvalue,val)


if __name__ == '__main__':
    main()
