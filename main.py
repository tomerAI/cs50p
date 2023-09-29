"""
A stock terminal that fetches stock data,
transform and present the data,
predict stock prices based on ML

"""

import sys
import yfinance as yf
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_score


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
            "Income Statement----------------------------------type 'is'\n" \
            "Cash Flow Statement-------------------------------type 'cf'\n" \
            "Predict if Price Increases Tomorrow---------------type 'pp'\n" \
            "Exit Program--------------------------------------type 'x'")
    options = ["is", "cf", "pp", "x"]
    option = str()
    while option not in options:
        option = input("Input: ")
    
    #income statement, cf, balance sheet
    if option == 'is':
        income_statement(ticker)

    elif option == 'cf':
        cashflow(ticker)
    
    #forecast trend
    elif option =='ft':
        forecast(ticker)

    #predict price
    elif option == 'pp':
        predict(ticker)

    #exit
    elif option == 'x':
        sys.exit()
    

def cashflow(ticker):
    symbol = yf.Ticker(ticker)
    cashflow_df = pd.DataFrame(symbol.cash_flow).transpose()
    cf_summary_df = cashflow_df[['Free Cash Flow', 'Operating Cash Flow', 'Investing Cash Flow', 'Financing Cash Flow']].transpose()
    print(cf_summary_df)


def income_statement(ticker):
    symbol = yf.Ticker(ticker)
    income_stmt_df = pd.DataFrame(symbol.income_stmt).transpose()
    income_stmt_sum_df = income_stmt_df[['Total Revenue', 'Gross Profit', 'EBITDA', 'EBIT']].transpose()
    print(income_stmt_sum_df)


def predict(ticker):
    #load data
    symbol = yf.Ticker(ticker)
    data = symbol.history(period="20y")
    #Fix date index format
    data.index = pd.to_datetime(data.index).date
    #create tomorrow column that is the close of the next day
    data['Tomorrow'] = data["Close"].shift(-1)
    #create target column that is 1 if the stock goes up the next day or 0 if opposite
    data['Target'] = (data['Tomorrow'] > data['Close']).astype(int)

    #Adding rolling mean as a predictor
    horizons = [2,5,60,250,1000]

    #adding predictors to data column
    for horizon in horizons:
        rolling_avg = data.rolling(horizon).mean()
        #print(rolling_avg)
        #Adding ratio_columns of current stock-price and previous stock-price
        ratio_column = f"Close_Ratio_{horizon}"    
        data[ratio_column] = data["Close"] / rolling_avg["Close"]
        
        #Adding a trend column
        trend_column = f"Trend_{horizon}"
        data[trend_column] = data.shift(1).rolling(horizon).sum()["Target"]

    data = data.dropna(axis=1, how="all")

    #predictors
    predictors = ['Close', 'Volume', 'Open', 'High', 'Low']
    for horizon in horizons:
        predictors += [f'Close_Ratio_{horizon}', f'Trend_{horizon}']

    # Filtering out predictors not present in the `data` DataFrame
    valid_predictors = [predictor for predictor in predictors if predictor in data.columns]

    #training sets
    num_rows = len(data)
    split_index = int(num_rows * 0.75)
    train = data.iloc[:split_index]
    test = data.iloc[split_index:]

    #model
    model = HistGradientBoostingClassifier(learning_rate=0.1,
                                        min_samples_leaf=20,
                                        random_state=1)

    #Training
    model.fit(train[valid_predictors], train["Target"])

    #Testing predictions on test data
    predictions = model.predict_proba(test[valid_predictors])[:,1]
    predictions[predictions >= .6] = 1
    predictions[predictions < .6] = 0
    predictions = pd.Series(predictions, index=test.index, name="Predictions")

    #Printing tomorrows result
    score = precision_score(test["Target"], predictions) * 100
    print(f"The model ran data from the previous 20 years and backtested 25% of the data with a score of {score:.2f}%")
    if predictions[-1] == 1.0:
        print("The model predicts the price to increase tomorrow with a confidence of 60%")
    else:
        print("The model doesn't predict the stock to increase tomorrow at a threshold of 60%")


if __name__ == "__main__":
    main()

