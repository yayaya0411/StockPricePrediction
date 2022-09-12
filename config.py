
# folder setting
gd_root = '.'
stock_root = 'stock'
assets_root = 'assets'
model_root = 'model'
data_root = 'data'
ror_root = 'ror'
assets_root = 'assets'

# model data setting
time_slide = 60
days = 3
ratio = 0.01

epochs = 200
batch_size = 256

price = ['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']
# price = ['Close','Adj Close']
indicators = ['CMO', 'RSI', 'MACD', 'WILLR', 'slowk', 'slowd']
margin = [   
    'MarginPurchaseBuy', 'MarginPurchaseCashRepayment','MarginPurchaseSell', 'MarginPurchaseTodayBalance', 
    'MarginPurchaseYesterdayBalance', 'OffsetLoanAndShort','ShortSaleBuy', 'ShortSaleSell', 
    'ShortSaleTodayBalance', 'ShortSaleYesterdayBalance', 
    'b_total', 's_total','b_s_ratio', 'bs_ratio', 'buy_volume', 'sell_volume', 
]

feature = margin + price #+ indicators

Update = False


# stock setting
# Y109Q2
# stockNum = ["1101", "1102", "1216", "1301", "1326", "1402", "2002", "2105", "2207", 
#             "2301", "2303", "2308", "2317", "2327", "2330", "2352", "2357", "2382", 
#             "2395", "2408", "2412", "2454", "2474", "2609", "2610", "2801", #"2633",
#             "2823", "2880", "2881", "2882", "2883", "2884", "2885", "2886", "2887", 
#             "2888", "2890", "2891", "2892", "2912", "3008", "3045", "4904", #"3711",
#             "4938",  "5880", "6505", "9904", "9910"] #"5871",
# Y110Q3
stockNum = ["1101", "1216", "1301", "1303", "1326", "1402", "1590", "2002", "2207", "2303", 
            "2308", "2317", "2324", "2327", "2330", "2357", "2379", "2382", "2395", #"2311",
            "2408", "2409", "2412", "2603", "2609", "2615", "2801", "2880", "2881", "2882", 
            "2884", "2885", "2886", "2887", "2891", "2892", "2912", "3008", "3034", "3045", 
            "4904", "4938", "5880", "6415", "6505", "8046", "8454", "9910"] 

# electronic
# stockNum = ['2301', '2303', '2308', '2317', '2327', '2330', '2352', '2357', '2382', '2395', '2408', '2454', '2474', '3008', '4938']

#finance
# stockNum = ['2801', '2823', '2880', '2881', '2882', '2883', '2884', '2885', '2886', '2887', '2888', '2890', '2891', '2892', '5880']

# stock period setting
download_start = '2000/01/01'
download_end = '2021/10/31'

train_start = '2013-01-01'
train_end = '2017-12-31'

test_start = '2018/01/01'
test_end = '2019/12/31'