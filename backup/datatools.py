import numpy as np
import pandas as pd
import pickle
import pandas_datareader as pdr
import datetime
import requests
import math
import os

def status(time_slide, feature):
    print(f'slide winodw: {time_slide} days, {len(feature)} features\n')

def print_time(colab):
    now = datetime.datetime.now()
    if colab:
        now += datetime.timedelta(hours=8)
    print(now.strftime('%Y/%m/%d %H:%M:%S'))

def finmind_dict(dataset_list):
    def get(url, parameter):
        translation = requests.get(url, params=parameter)
        trans = translation.json()
        return pd.DataFrame(trans['data'])

    for data in dataset_list:
        if data == 'margin':
            url = "https://api.finmindtrade.com/api/v4/translation" #load in三大法人投資情況
            parameter = {
                "dataset": "TaiwanStockMarginPurchaseShortSale"
            }
            print(get(url,parameter))

        if data == 'institution':
            url = "https://api.finmindtrade.com/api/v4/translation" #load in三大法人投資情況
            parameter = {
                "dataset": "TaiwanStockInstitutionalInvestorsBuySell",
            }
            print(get(url,parameter))

        if data == 'info':
            url = "https://api.finmindtrade.com/api/v4/data"
            parameter = {
                "dataset": "TaiwanStockInfo",
            }
            print(get(url,parameter))

def exponential_smoothing(days, alpha, numdata):
    # assign weights
    weights = []
    i = days 
    while i>0:
        weights.append( math.exp(-alpha*i) )
        i-=1
    weights = np.array(weights) / sum(weights)
    # return data after smoothing
    results = []
    results += numdata[:days]
    for i in range(days, len(numdata)):
        results.append((np.dot( np.array(weights), np.array(numdata[i-days:i]))))
    return results

def to_ratio(df):
    for col in df.columns:
        if col == 'Close': continue
        df[col] = df[col].shift(-1)/df[col]
    df = df.fillna(0).replace([np.inf, -np.inf], 0)    
    df = df.iloc[:-1]
    return df    

def to_onelist(text):
    classes =['0', '1']
    onehot = [0] * 2
    onehot[int(classes.index(str(text)))] = 1
    label_list = np.array(onehot)
    return label_list

def y_one_hot(y_list):
    y = []
    for i in range(len(y_list)):
        y.append(to_onelist(str(y_list[i])))
    return np.array(y)

def count_label_dist(arr):
    count_1 = 0 
    count_0 = 0 
    for i in range(len(arr)):
        if list(arr[i])== list([0,1]): count_1 += 1
        elif list(arr[i])== list([1,0]): count_0 += 1
    print(count_0, count_1)
    return count_0, count_1
    
def save_pickle(var, path):
    with open( path,'wb') as f:
        pickle.dump(var, f)

def load_pickle(path):
    with open(path,'rb') as f:
        x = pickle.load(f) 
    return x

def loading_data_api(stock, start, end):
    s = start.split('/')
    e = end.split('/')
    start = datetime.datetime(int(s[0]), int(s[1]), int(s[2]))
    end = datetime.datetime(int(e[0]), int(e[1]), int(e[2]))

    # Loading price data from yahoo api 
    Price_df = pdr.DataReader(str(stock)+'.TW', 'yahoo', start=start, end=end)
    Return_Test_Stock_df = Price_df.copy(deep=True)
    Price_df.index = Price_df.index.format(formatter=lambda x: x.strftime('%Y-%m-%d'))
    stock_time = str(Price_df.index[0]).split(" ")[0]
    
    # Loading Margin data from FinMind api 
    url = "https://api.finmindtrade.com/api/v3/data" #load in 融資融券
    parameter = {
        "dataset": "TaiwanStockMarginPurchaseShortSale",
        "stock_id": str(stock),
        "date": stock_time,
    }
    data = requests.get(url, params=parameter)
    data = data.json()
    data = pd.DataFrame(data['data'])
    data['b_total'] = data['MarginPurchaseLimit'] - data['MarginPurchaseTodayBalance']
    data['s_total'] = data['ShortSaleLimit'] - data['ShortSaleTodayBalance']
    data['b_s_ratio'] = (data['ShortSaleLimit'] - data['ShortSaleTodayBalance'])/(data['MarginPurchaseLimit'] - data['MarginPurchaseTodayBalance']+1)
    data = data.set_index('date')

    # 合併這兩個表格
    new_df = Price_df.join(data)
    new_df = new_df.drop(['stock_id'], axis=1)
    new_df = new_df.drop(['ShortSaleCashRepayment'], axis=1)
    new_df = new_df.drop(['MarginPurchaseLimit'], axis=1)
    new_df = new_df.drop(['ShortSaleLimit'], axis=1)
    
    # Loading Institutiona data from FinMind api 
    url = "https://api.finmindtrade.com/api/v3/data" #load in三大法人投資情況
    parameter = {
        "dataset": "InstitutionalInvestorsBuySell",
        "stock_id": str(stock),
        "date": stock_time,
    }
    data = requests.get(url, params=parameter)
    data = data.json()
    data = pd.DataFrame(data['data'])

    df = pd.DataFrame()
    z = 0 
    i = 0
    while i < len(data)  :
        a = data['date'][i]
        count = 0
        sell = 0
        buy = 0 
        if i+6 < len(data)-1:
            for j  in range(i, i+6):
                if data['date'][j]== a :
                    count += 1
                    sell += data['sell'][j]
                    buy += data['buy'][j]
        else:
            for j in range(i, len(data)):
                if data['date'][j] == a :
                    count += 1
                    sell += data['sell'][j]
                    buy += data['buy'][j]
        df = df.append({'date': a, 'buy_volume': buy, 'sell_volume':sell, 'bs_ratio':buy/(sell+1)}, ignore_index=True)
        i+=count

    new_df = new_df.join(df.set_index('date'))
    new_df = new_df.fillna(axis=0,method='ffill')
    new_df = new_df.drop(['Note'], axis=1)
    return new_df

def y_label(df):
    days = 5
    ans = [0]*days
    for i in range(days,len(df)):
        Pi = df['Close'][i]
        Pj = df['Close'][i-days]
        if Pi > Pj: 
            y = 1
        else: 
            y = 0
        ans.append(y)    
    df['y_label_before']= ans
    df['y_label_sum'] = df['y_label_before'].rolling(days).sum()
    df = df.iloc[days:,:]
    return df

def y_label_2(df, days, ratio):
    ans = [0]*days
    for i in range(days,len(df)):
        if i+days >= len(df):
            y = None
            ans.append(y)    
        else:    
            Pi = df['Close'][i]
            Pj = df['Close'][i+days]
            # print(i,Pi,Pi*1.01,Pj)
            if Pi * (1 + ratio) < Pj: 
                y = '1'
            else: 
                y = '0'
            ans.append(y)    
    df['y_label'] = ans
    df = df.iloc[:-days,:]
    return df

def y_price(df, days):
    ans = [0]*days
    for i in range(days,len(df)):
        if i+days >= len(df):
            y = None
            ans.append(y)
        else:
            y = df['Close'][i+days]
            ans.append(y)
    df['y_label'] = ans
    df = df.iloc[:-days, :]
    df_y = df['y_label']
    df = df.drop(['y_label'],axis=1)
    return df, df_y

def y_ratio(df,days):
    df['y_next'] = df['Close'].shift(-days)
    df = df.iloc[:-days]
    df['y_ratio'] = df['y_next']/df['Close']
    df_y = df['y_ratio']
    df = df.drop(['y_ratio','y_next'],axis=1)
    return df, df_y

def label(df,days):
    df['y_label'] = df['Close'].shift(-days)
    df['y_label'] = np.where(df['y_label']>df['Close'],1,0)
    df = df.iloc[:-days]
    df_y = df[['y_label']]
    df = df.drop(['y_label'],axis=1)
    return df, df_y

# Caculate n Days SMA
def sma(df,n,y):
    ma  = df.iloc[:,:-1].rolling(n).mean()
    ma_df = pd.concat([ma,df[y]],axis=1)
    ma_df = ma_df.iloc[n-1:,:]
    return ma_df

# Caculate n Days EMA
def ema(df,n,y):
    ma  = df.ewm(span=n).mean()
    ma_df = pd.concat([ma,df[y]],axis=1)
    ma_df = ma_df.iloc[n-1:,:]
    return ma_df

def feature_std(stockNum, path):
    df_stock = pd.DataFrame()
    df_std = pd.DataFrame()
    filter = False
    if filter:
        for stock_num in stockNum:
            print('', stock_num, end ='\r')
            df = pd.read_csv(os.path.join(path, stock_num+'.csv'),index_col=0)
            df_describe = df.describe()
            df_describe = df_describe[df_describe.index=='std']
            df_describe['stock'] = str(stock_num)
            df_std = pd.concat([df_std,df_describe])

            df['stock'] = str(stock_num)
            df_stock = pd.concat([df_stock, df])
        df_std = df_std.set_index('stock')
        feature_rank = df_std.mean().sort_values(ascending = False)
        feature_rank = pd.DataFrame(feature_rank,columns = ['std'])
        feature_rank['rank'] = feature_rank.rank(ascending=False).astype('int')
        # feature_rank.to_csv(os.path.join(gd_root,assets_root,'feature_rank.csv'))
    return feature_rank

def input_build(new_x, y, ori_y):
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    # val_x = []
    # val_y = []
    
    tr_va = int(len(new_x)*8/10)
    y = list(y)

    # original add all data into training
    train_x = train_x + new_x[:tr_va]
    train_y = train_y + y[:tr_va]
    # val_x = val_x + new_x[tr_va: va_te]
    # val_y = val_y + y[tr_va: va_te]
    test_x = test_x + new_x[tr_va:]
    test_y = test_y +y[tr_va:]

    # for test scaler y
    ori_y_train = ori_y[:tr_va]
    ori_y = ori_y[tr_va:]

    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y), ori_y_train, ori_y

def normalization(df):
    for col in df.columns:
        if col == 'y_label' or col =='bs_ratio' or col =='b_s_ratio': continue
        mean = df[str(col)].mean()
        stdev = df[str(col)].std()
        if stdev == 0: 
            stdev = 1
        df[str(col)] = df[str(col)].apply(lambda x : (x-mean)/stdev)
    return df

def minmax(df):
    for col in df.columns:
        if col == 'y_label' or col =='bs_ratio' or col =='b_s_ratio': continue
        max = df[str(col)].max()
        min = df[str(col)].min()
        if min == 0: 
            min = 1
        df[str(col)] = df[str(col)].apply(lambda x : (x-min)/(max-min))
    return df    
