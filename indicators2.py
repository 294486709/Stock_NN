import pandas as pd
import numpy as np




def SMA_bookaa(d, N):
    v = pd.Series(index=d.index)
    last = np.nan
    for key in d.index:
        x = d[key]
        if last == last:
            x1 = (x + (N - 1) * last) / N
        else:
            x1 = x
        last = x1
        v[key] = x1
        if x1 != x1:
            last = x
    return v

def zhibiao_kdj(data, N1=9, N2=3, N3=3):
    low1 = pd.rolling_min(data.Low, N1)
    high1 = pd.rolling_max(data.High, N1)
    rsv = (data.Close - low1) / (high1 - low1) * 100
    k = SMA_bookaa(rsv,N2)
    d = SMA_bookaa(k, N3)
    j = k * 3 - d * 2
    return k,d,j

def adx(data, n=14, fillna=False):
    high = data.High
    low = data.Low
    close = data.Close
    cs = close.shift(1)
    tr = high.combine(cs, max) - low.combine(cs, min)
    trs = tr.rolling(n).sum()
    up = high - high.shift(1)
    dn = low.shift(1) - low
    pos = ((up > dn) & (up > 0)) * up
    neg = ((dn > up) & (dn > 0)) * dn
    dip = 100 * pos.rolling(n).sum() / trs
    din = 100 * neg.rolling(n).sum() / trs
    dx = 100 * np.abs((dip - din)/(dip + din))
    adx = dx.ewm(n).mean()
    if fillna:
        adx = adx.fillna(40)
    return pd.Series(adx, name='adx')


def adx_pos(data, n=14, fillna=False):
    high = data.High
    low = data.Low
    close = data.Close
    cs = close.shift(1)
    tr = high.combine(cs, max) - low.combine(cs, min)
    trs = tr.rolling(n).sum()
    up = high - high.shift(1)
    dn = low.shift(1) - low
    pos = ((up > dn) & (up > 0)) * up
    neg = ((dn > up) & (dn > 0)) * dn
    dip = 100 * pos.rolling(n).sum() / trs
    if fillna:
        dip = dip.fillna(20)
    return pd.Series(dip, name='adx_pos')


def adx_neg(data, n=14, fillna=False):
    high = data.High
    low = data.Low
    close = data.Close
    cs = close.shift(1)
    tr = high.combine(cs, max) - low.combine(cs, min)
    trs = tr.rolling(n).sum()
    up = high - high.shift(1)
    dn = low.shift(1) - low
    pos = ((up > dn) & (up > 0)) * up
    neg = ((dn > up) & (dn > 0)) * dn
    din = 100 * neg.rolling(n).sum() / trs
    if fillna:
        din = din.fillna(20)
    return pd.Series(din, name='adx_neg')


def adx_indicator(data, n=14, fillna=False):
    high = data.High
    low = data.Low
    close = data.Close
    cs = close.shift(1)
    tr = high.combine(cs, max) - low.combine(cs, min)
    trs = tr.rolling(n).sum()
    up = high - high.shift(1)
    dn = low.shift(1) - low
    pos = ((up > dn) & (up > 0)) * up
    neg = ((dn > up) & (dn > 0)) * dn
    dip = 100 * pos.rolling(n).sum() / trs
    din = 100 * neg.rolling(n).sum() / trs
    adx_diff = dip - din
    # prepare indicator
    df = pd.DataFrame([adx_diff]).T
    df.columns = ['adx_diff']
    df['adx_ind'] = 0
    df.loc[df['adx_diff'] > 0, 'adx_ind'] = 1
    adx_ind = df['adx_ind']

    if fillna:
        adx_ind = adx_ind.fillna(0)
    return pd.Series(adx_ind, name='adx_ind')


def rsi(data, n=14, fillna=False):
    close = data.Close
    diff = close.diff()
    which_dn = diff < 0
    up, dn = diff, diff*0
    up[which_dn], dn[which_dn] = 0, -up[which_dn]
    emaup = up.ewm(n).mean()
    emadn = dn.ewm(n).mean()
    rsi = 100 * emaup/(emaup + emadn)
    if fillna:
        rsi = rsi.fillna(50)
    return pd.Series(rsi, name='rsi')


def cci(data, n=20, c=0.015, fillna=False):
    high = data.High
    low = data.Low
    close = data.Close
    pp = (high+low+close)/3
    cci = (pp-pp.rolling(n).mean())/pp.rolling(n).std()
    cci = 1/c * cci
    if fillna:
        cci = cci.fillna(0)
    return pd.Series(cci, name='cci')


def bollinger_mavg(data, n=20, fillna=False):
    close = data.Close
    mavg = close.rolling(n).mean()
    if fillna:
        mavg = mavg.fillna(method='backfill')
    return pd.Series(mavg, name='mavg')


def bollinger_hband(data, n=20, ndev=2, fillna=False):
    close = data.Close
    mavg = close.rolling(n).mean()
    mstd = close.rolling(n).std()
    hband = mavg + ndev*mstd
    if fillna:
        hband = hband.fillna(method='backfill')
    return pd.Series(hband, name='hband')


def bollinger_lband(data, n=20, ndev=2, fillna=False):
    close = data.Close
    mavg = close.rolling(n).mean()
    mstd = close.rolling(n).std()
    lband = mavg - ndev*mstd
    if fillna:
        lband = lband.fillna(method='backfill')
    return pd.Series(lband, name='lband')


def bollinger_hband_indicator(data, n=20, ndev=2, fillna=False):
    close = data.Close
    df = pd.DataFrame([close]).transpose()
    mavg = close.rolling(n).mean()
    mstd = close.rolling(n).std()
    hband = mavg + ndev*mstd
    df['hband'] = 0.0
    df.loc[close > hband, 'hband'] = 1.0
    hband = df['hband']
    if fillna:
        hband = hband.fillna(0)
    return pd.Series(hband, name='bbihband')


def bollinger_lband_indicator(data, n=20, ndev=2, fillna=False):
    close = data.Close
    df = pd.DataFrame([close]).transpose()
    mavg = close.rolling(n).mean()
    mstd = close.rolling(n).std()
    lband = mavg - ndev*mstd
    df['lband'] = 0.0
    df.loc[close < lband, 'lband'] = 1.0
    lband = df['lband']
    if fillna:
        lband = lband.fillna(0)
    return pd.Series(lband, name='bbilband')


def acc_dist_index(data, fillna=False):
    high = data.High
    low = data.Low
    close = data.Close
    volume = data.Volume
    clv = ((close - low) - (high - close)) / (high - low)
    clv = clv.fillna(0.0) # float division by zero
    ad = clv * volume
    ad = ad + ad.shift(1)
    if fillna:
        ad = ad.fillna(0)
    return pd.Series(ad, name='adi')



def on_balance_volume(data, fillna=False):
    close = data.Close
    volume = data.Volume
    df = pd.DataFrame([close, volume]).transpose()
    df['OBV'] = 0
    c1 = close < close.shift(1)
    c2 = close > close.shift(1)
    if c1.any():
        df.loc[c1, 'OBV'] = - volume
    if c2.any():
        df.loc[c2, 'OBV'] = volume
    obv = df['OBV']
    if fillna:
        obv = obv.fillna(0)
    return pd.Series(obv, name='obv')


def on_balance_volume_mean(data, n=10, fillna=False):
    close = data.Close
    volume = data.Volume
    df = pd.DataFrame([close, volume]).transpose()
    df['OBV'] = 0
    c1 = close < close.shift(1)
    c2 = close > close.shift(1)
    if c1.any():
        df.loc[c1, 'OBV'] = - volume
    if c2.any():
        df.loc[c2, 'OBV'] = volume
    obv = df['OBV'].rolling(n).mean()
    if fillna:
        obv = obv.fillna(0)
    return pd.Series(obv, name='obv')


def chaikin_money_flow(data, n=20, fillna=False):
    high = data.High
    close = data.Close
    low =  data.Low
    close = data.Close
    volume = data.Volume
    mfv = ((close - low) - (high - close)) / (high - low)
    mfv = mfv.fillna(0.0) # float division by zero
    mfv *= volume
    cmf = mfv.rolling(n).sum() / volume.rolling(n).sum()
    if fillna:
        cmf = cmf.fillna(0)
    return pd.Series(cmf, name='cmf')


def force_index(data, n=2, fillna=False):
    close = data.Close
    volume = data.Volume
    fi = close.diff(n) * volume.diff(n)
    if fillna:
        fi = fi.fillna(0)
    return pd.Series(fi, name='fi_'+str(n))


def ease_of_movement(data, n=20, fillna=False):
    high = data.High
    low = data.Low
    volume = data.Volume
    emv = (high.diff(1) + low.diff(1)) * (high - low) / (2 * volume)
    emv = emv.rolling(n).mean()
    if fillna:
        emv = emv.fillna(0)
    return pd.Series(emv, name='eom_' + str(n))


def volume_price_trend(data, fillna=False):
    close = data.Close
    volume = data.Volume
    vpt = volume * ((close - close.shift(1)) / close.shift(1))
    vpt = vpt.shift(1) + vpt
    if fillna:
        vpt = vpt.fillna(0)
    return pd.Series(vpt, name='vpt')


def average_true_range(data, n=14, fillna=False):
    high = data.High
    low = data.Low
    close = data.Close
    cs = close.shift(1)
    tr = high.combine(cs, max) - low.combine(cs, min)
    tr = tr.ewm(n).mean()
    if fillna:
        tr = tr.fillna(0)
    return pd.Series(tr, name='atr')


def keltner_channel_hband(data, n=10, fillna=False):
    high = data.High
    low = data.Low
    close = data.Close
    tp = ((4*high) - (2*low) + close) / 3.0
    tp = tp.rolling(n).mean()
    if fillna:
        tp = tp.fillna(method='backfill')
    return pd.Series(tp, name='kc_hband')


def keltner_channel_lband(data, n=10, fillna=False):
    high = data.High
    low = data.Low
    close = data.Close
    tp = ((-2*high) + (4*low) + close) / 3.0
    tp = tp.rolling(n).mean()
    if fillna:
        tp = tp.fillna(method='backfill')
    return pd.Series(tp, name='kc_lband')


def donchian_channel_hband(data, n=20, fillna=False):
    close = data.Close
    hband = close.rolling(n).max()
    if fillna:
        hband = hband.fillna(method='backfill')
    return pd.Series(hband, name='dchband')


def donchian_channel_lband(data, n=20, fillna=False):
    close = data.Close
    lband = close.rolling(n).min()
    if fillna:
        lband = lband.fillna(method='backfill')
    return pd.Series(lband, name='dclband')



def vortex_indicator_pos(data, n=14, fillna=False):
    high = data.High
    low = data.Low
    close = data.Close
    tr = high.combine(close.shift(1), max) - low.combine(close.shift(1), min)
    trn = tr.rolling(n).sum()
    vmp = np.abs(high - low.shift(1))
    vmm = np.abs(low - high.shift(1))
    vip = vmp.rolling(n).sum() / trn
    if fillna:
        vip = vip.fillna(1)
    return pd.Series(vip, name='vip')


def vortex_indicator_neg(data, n=14, fillna=False):
    high = data.High
    close = data.Close
    low = data.Low
    tr = high.combine(close.shift(1), max) - low.combine(close.shift(1), min)
    trn = tr.rolling(n).sum()

    vmp = np.abs(high - low.shift(1))
    vmm = np.abs(low - high.shift(1))

    vin = vmm.rolling(n).sum() / trn
    if fillna:
        vin = vin.fillna(1)
    return pd.Series(vin, name='vin')


def trix(data, n=15, fillna=False):
    close = data.Close
    ema1 = close.ewm(span=n, min_periods=n-1).mean()
    ema2 = ema1.ewm(span=n, min_periods=n-1).mean()
    ema3 = ema2.ewm(span=n, min_periods=n-1).mean()
    trix = (ema3 - ema3.shift(1)) / ema3.shift(1)
    trix *= 100
    if fillna:
        trix = trix.fillna(0)
    return pd.Series(trix, name='trix_'+str(n))


def mass_index(data, n=9, n2=25, fillna=False):
    high = data.High
    low = data.Low
    amplitude = high - low
    ema1 = amplitude.ewm(span=n, min_periods=n-1).mean()
    ema2 = ema1.ewm(span=n, min_periods=n-1).mean()
    mass = ema1/ema2
    mass = mass.rolling(n2).sum()
    if fillna:
        mass = mass.fillna(n2)
    return pd.Series(mass, name='mass_index_'+str(n))


def dpo(data, n=20, fillna=False):
    close = data.Close
    dpo = close.shift(int(n/(2+1))) - close.rolling(n).mean()
    if fillna:
        dpo = dpo.fillna(0)
    return pd.Series(dpo, name='dpo_'+str(n))


def kst(data, r1=10, r2=15, r3=20, r4=30, n1=10, n2=10, n3=10, n4=15, fillna=False):
    close = data.Close
    rocma1 = ((close - close.shift(r1)) / close.shift(r1)).rolling(n1).mean()
    rocma2 = ((close - close.shift(r2)) / close.shift(r2)).rolling(n2).mean()
    rocma3 = ((close - close.shift(r3)) / close.shift(r3)).rolling(n3).mean()
    rocma4 = ((close - close.shift(r4)) / close.shift(r4)).rolling(n4).mean()
    kst = 100*(rocma1 + 2*rocma2 + 3*rocma3 + 4*rocma4)
    if fillna:
        kst = kst.fillna(0)
    return pd.Series(kst, name='kst')


def kst_sig(data, r1=10, r2=15, r3=20, r4=30, n1=10, n2=10, n3=10, n4=15, nsig=9, fillna=False):
    close = data.Close
    rocma1 = ((close - close.shift(r1)) / close.shift(r1)).rolling(n1).mean()
    rocma2 = ((close - close.shift(r2)) / close.shift(r2)).rolling(n2).mean()
    rocma3 = ((close - close.shift(r3)) / close.shift(r3)).rolling(n3).mean()
    rocma4 = ((close - close.shift(r4)) / close.shift(r4)).rolling(n4).mean()
    kst = 100*(rocma1 + 2*rocma2 + 3*rocma3 + 4*rocma4)
    kst_sig = kst.rolling(nsig).mean()
    if fillna:
        kst_sig = kst_sig.fillna(0)
    return pd.Series(kst_sig, name='kst_sig')


def ichimoku_a(data, n1=9, n2=26, fillna=False):
    low = data.Low
    high = data.High
    conv = (high.rolling(n1).max() + low.rolling(n1).min()) / 2
    base = (high.rolling(n2).max() + low.rolling(n2).min()) / 2

    spana = (conv + base) / 2
    spana = spana.shift(n2)
    if fillna:
        spana = spana.fillna(method='backfill')
    return pd.Series(spana, name='ichimoku_a_'+str(n2))


def ichimoku_b(data, n2=26, n3=52, fillna=False):
    high = data.High
    low = data.Low
    spanb = (high.rolling(n3).max() + low.rolling(n3).min()) / 2
    spanb = spanb.shift(n2)
    if fillna:
        spanb = spanb.fillna(method='backfill')
    return pd.Series(spanb, name='ichimoku_b_'+str(n2))


def money_flow_index(data, n=14, fillna=False):
    high = data.High
    low = data.Low
    close = data.Close
    volume = data.Volume
    # 0 Prepare dataframe to work
    df = pd.DataFrame([high, low, close, volume]).T
    df.columns = ['High', 'Low', 'Close', 'Volume']
    df['Up_or_Down'] = 0
    df.loc[(df['Close'] > df['Close'].shift(1)), 'Up_or_Down'] = 1
    df.loc[(df['Close'] < df['Close'].shift(1)), 'Up_or_Down'] = 2

    # 1 typical price
    tp = (df['High'] + df['Low'] + df['Close']) / 3.0

    # 2 money flow
    mf = tp * df['Volume']

    # 3 positive and negative money flow with n periods
    df['1p_Positive_Money_Flow'] = 0.0
    df.loc[df['Up_or_Down'] == 1, '1p_Positive_Money_Flow'] = mf
    n_positive_mf = df['1p_Positive_Money_Flow'].rolling(n).sum()

    df['1p_Negative_Money_Flow'] = 0.0
    df.loc[df['Up_or_Down'] == 2, '1p_Negative_Money_Flow'] = mf
    n_negative_mf = df['1p_Negative_Money_Flow'].rolling(n).sum()

    # 4 money flow index
    mr = n_positive_mf / n_negative_mf
    mr = (100 - (100 / (1 + mr)))
    if fillna:
        mr = mr.fillna(50)
    return pd.Series(mr, name='mfi_'+str(n))


def tsi(data, r=25, s=13, fillna=False):
    close = data.Close
    m = close - close.shift(1)
    m1 = m.ewm(r).mean().ewm(s).mean()
    m2 = abs(m).ewm(r).mean().ewm(s).mean()
    tsi = m1/m2
    tsi *= 100
    if fillna:
        tsi = tsi.fillna(0)
    return pd.Series(tsi, name='tsi')


def daily_return(data, fillna=False):
    close = data.Close
    dr = (close / close.shift(1)) - 1
    dr *= 100
    if fillna:
        dr = dr.fillna(0)
    return pd.Series(dr, name='d_ret')


def cumulative_return(data, fillna=False):
    close = data.Close
    cr = (close / close.iloc[0]) - 1
    cr *= 100
    if fillna:
        cr = cr.fillna(method='backfill')
    return pd.Series(cr, name='cum_ret')



def pandas_to_list(data):
    keys = data.keys()
    newlist = [data[i].tolist() for i in keys]
    return newlist