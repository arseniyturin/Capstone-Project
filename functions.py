import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import plotting


z_disribution = lambda x: (x - x.mean()) / x.std() # works as a map function or in list comprehension
norm = lambda x: ( x - x.min() ) / ( x.max() - x.min() ) # works as a map function or in list comprehension
gain = lambda x: x if x > 0 else 0 # works as a map function or in list comprehension
loss = lambda x: abs(x) if x < 0 else 0 # works as a map function or in list comprehension

binary = lambda data: [1 if x > 0 else 0 for x in data]
        
# split a univariate sequence into samples
def split_sequence(sequence, n_steps, split=True, ratio=0.8):
    X, y = [], []
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    if split == True:
        # X_train, y_train, X_test, y_test
        return np.array(X[:round(len(X)*ratio)]), np.array(y[:round(len(X)*ratio)]), np.array(X[round(len(X)*ratio):]), np.array(y[round(len(X)*ratio):])
    else:
        return np.array(X), np.array(y)

# split a multivariate sequence into samples
def split_sequences(features, target, n_steps, split=True, ratio=0.8):
    X, y = [], []
    for i in range(len(features)):
        end_ix = i + n_steps
        if end_ix > len(features):
            break
        seq_x, seq_y = features[i:end_ix], target[end_ix-1]
        X.append(seq_x)
        y.append(seq_y)
    if split == True:
        # X_train, y_train, X_test, y_test
        X_train = np.array(X[:round(len(X)*ratio)])
        y_train = np.array(y[:round(len(X)*ratio)])
        X_test  = np.array(X[round(len(X)*ratio):])
        y_test  = np.array(y[round(len(X)*ratio):])
        return X_train, y_train, X_test, y_test
    else:
        return np.array(X), np.array(y)    

# Evaluation of ML model
def evaluation(X, y, model, n_preds=10, random=True, show_graph=True):
      
    n_steps = X.shape[1]
    max_random_int = len(y) - n_steps
    y_true, y_pred, prediction_accuracy, slices = [], [], [], []
    
    for i in range(n_preds):        
        
        if random == True:          
            position = np.random.randint(0, max_random_int)
        else: 
            position = i
            
        y_hat = model.predict(X[position:position+1])[0][0]
        y_pred.append(y_hat)
        y_true.append(y[position])               
        y_current = y[position]
                
        # If we predit return, c = 0, else c = previous sequence position
        if y.min() < 0:
            c = 0
        else: 
            c = y[position-1]
        
        if ((y_hat > c) & (y_current > c)) or ((y_hat < c) & (y_current < c)): acc = 1           
        else: acc = 0
        
        prediction_accuracy.append(acc)       
        slices.append((list(y[position-n_steps:position+1]), list(y[position-n_steps:position]) + [y_hat], acc))
        
    if show_graph == True:
        plt.rcParams['figure.dpi'] = 227
        plt.style.use('seaborn-whitegrid')
        plt.figure(figsize=(16,6))
        plt.bar(range(n_preds), y_true[:], width=.7, alpha=.6, color="#4ac2fb", label="True")
        plt.bar(range(n_preds), y_pred[:], width=.7, alpha=.6, color="#ff4e97", label="Predicted")
        plt.axhline(0, color="#333333", lw=.8)
        plt.legend(loc=1)
        plt.title('Daily Return Prediction', fontSize=15)
        plt.show()
    
    print('MSE:', mean_squared_error(y_true, y_pred))
    print('Accuracy: {}%'.format( round((sum(prediction_accuracy)/len(prediction_accuracy))*100 ),2) )
    return slices, np.array(y_true), np.array(y_pred)
    
# Function to scale all features to [0-1]    
def scale(dataframe, scale=(0,1)):
    columns = dataframe.columns
    scaler = MinMaxScaler()
    scaler.feature_range = scale
    return pd.DataFrame(scaler.fit_transform(dataframe), columns=columns).dropna()
    
def bb_trading(stock, budget=15000, u=0.03, l=0.03, show_graph=True, show_return=True):
    
    money = budget
    stock = stock.reset_index()
    net = []
    for i in range(len(stock)):
        today = stock.iloc[i]
        
        # Buy
        if (today.Close < today.MA21) and (abs(1 - today.Close / today.Lower_band) < u):
            if money > 0:
                stock_amt = int(money / today.Close)
                money = 0
                net.append([today.Date, today.Close, 1, stock_amt * today.Close])

        # Sell        
        if (today.Close > today.MA21) and (abs(1 - today.Upper_band / today.Close) < l):
            if money == 0:
                money = stock_amt * today.Close
                stock_amt = 0
                net.append([today.Date, today.Close, 0, money])
        
    profit = net[-1][3] - budget
    
    if show_return == True:
        print('Number of Trades: {}'.format(len(net)))
        print('Time Frame: {} days'.format((net[-1][0] - net[0][0]).days))
        print('Profit: ${:.2f} | {}%'.format(profit, round(profit/budget*100, 2)))

    if show_graph == True:
        plotting.trading_history(stock.set_index('Date'), net)

    return profit, net
    
def macd_trading(stock, budget=15000, show_graph=True, show_return=True):
    
    state_check = lambda x, y: 1 if x > y else 0
    stock = stock.reset_index()    
    money = budget
    prev_state = state_check(stock.loc[0].MACD, stock.loc[0].Signal)
    stock_amt = 0
    net = []

    for i in range(1, len(stock)):

        today = stock.iloc[i]
        state = state_check(today.MACD, today.Signal)

        if state != prev_state:

            # BUY
            if (today.MACD > today.Signal) and (money != 0):
                stock_amt = int(money / today.Close)
                money = 0
                net.append([today.Date, today.Close, 1, stock_amt * today.Close])

            # SELL
            if (today.MACD < today.Signal) and (stock_amt != 0):
                money = stock_amt * today.Close
                stock_amt = 0
                net.append([today.Date, today.Close, 0, money])

        prev_state = state
    
    profit = net[-1][3] - budget
    
    if show_return == True:
        print('Number of Trades: {}'.format(len(net)))
        print('Time Frame: {} days'.format((net[-1][0] - net[0][0]).days))
        print('Profit: ${:.2f} | {}%'.format(profit, round(profit/budget*100, 2)))

    if show_graph == True:
        plotting.trading_history(stock.set_index('Date'), net)

    return profit, net    
    
'''
def rsi_slope(rsi):
    a = [0]
    for i in range(len(rsi)):
        if i+1 == len(rsi):
            break
        a.append( rsi[i] - rsi[i+1] )
    a = np.array(a)
    a = (a-a.mean())/(a.max()-a.min())
    return a
'''
