import pandas as pd
from sklearn import preprocessing
from functions import gain, loss


# Relative Strength Index
def rsi(stock):    
    # Create a list, fill first 14 values with 'None'
    rsi_list = [None for i in range(14)]
    # Change as an input
    stock = stock.Change
    
    # Calculating first RSI
    avg_gain = sum([i for i in stock[1:15] if i > 0])/14
    avg_loss = sum([abs(i) for i in stock[1:15] if i < 0])/14
    rs = avg_gain / avg_loss
    rsi = 100 - ( 100 / ( 1 + rs ))
    rsi_list.append(rsi)
    
    # Calculating following RSI's
    for i in range(15, len(stock)):
        avg_gain = (avg_gain * 13 + gain(stock[i]))/14
        avg_loss = (avg_loss * 13 + loss(stock[i]))/14
        rs = avg_gain / avg_loss
        rsi = 100 - ( 100 / ( 1 + rs ))
        rsi_list.append(rsi)
    
    return rsi_list   
 
# Moving Average Convergence/Divergence        
def macd(stock):
    exp1 = stock.Close.ewm(span=12, adjust=False).mean()
    exp2 = stock.Close.ewm(span=26, adjust=False).mean()
    macd = exp1-exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal
 
# Bollinger Bands    
def bollinger_bands(stock, window=21):
    rolling_mean = stock.Close.rolling(window).mean()
    rolling_std = stock.Close.rolling(window).std()
    upper_band = rolling_mean + (rolling_std*2)
    lower_band = rolling_mean - (rolling_std*2)
    return upper_band, lower_band
 
# Moving Average (7 days period)    
def ma7(stock):
    return stock.Close.rolling(7).mean()
  
# Moving Average (21 days period)  
def ma21(stock):
    return stock.Close.rolling(21).mean()
    
def momentum(data, n_days):
    m = [None for i in range(n_days)]    
    for i in range(len(data) - n_days):
        end = i + n_days
        m.append(data[i] - n_days)
    return m

####################################################
### Parse tesla news headlines from nasdaq.com
####################################################    
def get_tesla_headlines(page):
    html = requests.get(page).text
    soup = BeautifulSoup(html)    
    headlines = soup.find_all("a", { "target" : "_self" })
    headlines.pop(0)
    dates = soup.findAll('small')
    dates.pop(0)
    return [i.text.strip() for i in headlines], [i.text.strip().split()[0] for i in dates]