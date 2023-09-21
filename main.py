"""
A stock terminal that can load data from stocks through API,
transform the data,
create trend lines using forecasting,
predict stock prices based on different settings using ML

"""

import sys, requests, json
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

#load data
msft = yf.Ticker("MSFT")
data = msft.history(period="3y")
#Fix date index format
data.index = pd.to_datetime(data.index).date
#create tomorrow column that is the close of the next day
data['Tomorrow'] = data["Close"].shift(-1)
#create target column that is 1 if the stock goes up the next day or 0 if opposite
data['Target'] = (data['Tomorrow'] > data['Close']).astype(int)

#create model
model = RandomForestClassifier(n_estimators=100, 
                               min_samples_split=100, 
                               random_state=1)
#training sets
train = data.iloc[:-100]
test = data.iloc[-100:]
latest_day = data.iloc[-1:]

#predictors
predictors = ['Close', 'Volume', 'Open', 'High', 'Low']
model.fit(train[predictors], train["Target"])

#Predictions
predictions = pd.Series(model.predict(test[predictors]), index=test.index)
#Precision score: (target-predictions)
print(precision_score(test["Target"], predictions))
#combined = pd.concat([test["Target"], predictions], axis=1)
 
#Adding rolling mean as a predictor
horizons = [2,5,60,250,1000]
new_predictors = []

#adding predictors
for horizon in horizons:
    rolling_avg = data.rolling(horizon).mean()
    #print(rolling_avg)
    #Adding ratio_columns of current stock-price and previous stock-price
    ratio_column = f"Close_Ratio_{horizon}"    
    data[ratio_column] = data["Close"] / rolling_avg["Close"]
    
    #Adding a trend column
    trend_column = f"Trend_{horizon}"
    data[trend_column] = data.shift(1).rolling(horizon).sum()["Target"]

    #Adding new predictors
    new_predictors += [ratio_column, trend_column]

data = data.dropna(axis=1, how="all")

#model
model = RandomForestClassifier(n_estimators=200,
                               min_samples_split=50,
                               random_state=1)

#Training
model.fit(train[predictors], train["Target"])

#Predicting
predictions = model.predict_proba(test[predictors])[:,1]
predictions[predictions >= .6] = 1
predictions[predictions < .6] = 0
predictions = pd.Series(predictions, index=test.index, name="Predictions")
print(precision_score(test["Target"], predictions))

predict_next_day = model.predict(latest_day[predictors])
predictions[predictions >= .6] = 1
predictions[predictions < .6] = 0
predictions = pd.Series(predictions, index=test.index, name="Predictions")
print(predict_next_day[0])


def main():
    # TICKER VALIDATION
    valid_tickers = tickers()
    ticker = str()
    while ticker not in valid_tickers:
        ticker = input("Stock ticker: ").upper()
    print(f"{ticker} is valid")
    
    # MENU OPTIONS
    menu(ticker)


# SPY TICKERS
def tickers():
    ticker_list = []
    with open('constituents_csv.csv', 'r') as file:
        for line in file:
            symbol = line.rstrip().split(",")[0]
            ticker_list.append(symbol)
    return ticker_list


def menu(ticker):
    print(  "PICK YOUR NEXT ACTION\n" \
            "Income Statement------------   type 'is'\n" \
            "Cash Flow Statement---------   type 'cf'\n" \
            "Forecast Trend--------------   type 'fc'\n" \
            "Predict Price---------------   type 'pd'\n" \
            "Exit Program----------------   type 'x'")
    options = ["is", "cf", "bs", "fc", "pd", "x"]
    option = str()
    while option not in options:
        option = input("Input: ")
    
    #income statement, cf, balance sheet
    if option == 'is':
        income_statement(ticker)

    elif option == 'cf':
        cashflow(ticker)
    
    #forecast trend
    elif option =='fc':
        forecast(ticker)

    #predict price

    #exit
    elif option == 'x':
        sys.exit()
    

def cashflow(ticker):
    symbol = yf.Ticker(ticker)
    cashflow_df = pd.DataFrame(symbol.cash_flow).transpose()
    cf_summary_df = cashflow_df[['Free Cash Flow', 'Operating Cash Flow', 'Investing Cash Flow', 'Financing Cash Flow']].transpose()
    print(cf_summary_df)
    return menu(symbol)

def income_statement(ticker):
    symbol = yf.Ticker(ticker)
    income_stmt_df = pd.DataFrame(symbol.income_stmt).transpose()
    income_stmt_sum_df = income_stmt_df[['Total Revenue', 'Gross Profit', 'EBITDA', 'EBIT']].transpose()
    print(income_stmt_sum_df)
    return menu(symbol)


def forecast():
    ...

"""
def predict():
    ...
"""

if __name__ == "__main__":
    main()

