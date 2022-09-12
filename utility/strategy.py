
def simple_strategy(predict, own_stock, left_money, price):
    if predict == 1:  # 如果預測漲
        if own_stock == 0: ## 手上沒有股票
            own_stock += int( left_money / ( price * (1 + 0.001425)) ) #買股票
            left_money = left_money - own_stock*price*(1 + 0.001425) #計算剩餘金額
    elif predict == 0 :  # 如果預測不會漲
        if own_stock > 0: ## 手上有股票
            left_money += own_stock*price*(1 - 0.001425 - 0.003)  # 賣出股票
            own_stock = 0
    stock_assets = left_money + own_stock * price    
    return stock_assets, own_stock, left_money


def simple_strategy_vote(own_stock, left_money, price, recode, vote):
    days = sum(recode[-5:])
    if days >= vote:  # 如果預測漲
        if own_stock == 0: ## 手上沒有股票
            own_stock += int( left_money / ( price * (1 + 0.001425)) ) #買股票
            left_money = left_money - own_stock*price*(1 + 0.001425) #計算剩餘金額
    elif days < vote :  # 如果預測不會漲
        if own_stock > 0: ## 手上有股票
            left_money += own_stock*price*(1 - 0.001425 - 0.003)  # 賣出股票
            own_stock = 0
    stock_assets = left_money + own_stock * price    
    return stock_assets, own_stock, left_money    

def highlow_strategy(pred, own_stock, left_money, price ):
    if pred >= price:  # 5天後大於1天後
        if own_stock == 0: ## 手上沒有股票
            action = 'buy'
            own_stock += int( left_money / ( price * (1 + 0.001425)) ) #買股票
            left_money = left_money - own_stock * price *(1 + 0.001425) #計算剩餘金額
        else:
            action = 'hold'    
    elif pred < price :  # 如果預測不會漲
        if own_stock > 0: ## 手上有股票
            action = 'sell'
            left_money += own_stock*price*(1 - 0.001425 - 0.003)  # 賣出股票
            own_stock = 0
        else:
            action = 'keep'
    stock_assets = left_money + own_stock * price    
    return stock_assets, own_stock, left_money, action, pred, price 
