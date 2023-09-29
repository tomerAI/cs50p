    # Simple Terminal
    #### Video Demo:  https://youtu.be/odcnJdR5w84
    #### Description:

This is a terminal-based application that lets the user:

Fetch data from stock tickers using the Yahoo Finance API.
View specific financial statements like the income statement and cash flow statement.
Predict whether the stock price will increase tomorrow based on machine learning using historical data.


Dependencies:
sys: For system-specific parameters and functions.
yfinance: To fetch stock data.
pandas: For data manipulation and analysis.
sklearn.ensemble: For the machine learning model (HistGradientBoostingClassifier).
sklearn.metrics: To measure the precision score of the prediction.


Functions:
main(): The main function that validates the ticker input and displays the menu for user actions.
tickers(): Returns a list of valid tickers by reading them from 'constituents_csv.csv'.
menu(ticker): Displays the main menu where the user can choose to view financial statements, predict stock prices, or exit.
cashflow(ticker): Fetches and displays the cash flow statement of the specified ticker.
income_statement(ticker): Fetches and displays the income statement of the specified ticker.
predict(ticker): Loads the stock's historical data and uses a machine learning model to predict whether the stock price will increase the next day. It leverages rolling averages as features for the prediction.


Usage:
When run, the program will:
Ask the user for a stock ticker. The ticker is then validated against a list of valid tickers from 'constituents_csv.csv'.
Once a valid ticker is entered, the program will display a menu with the following options:
View Income Statement by typing 'is'.
View Cash Flow Statement by typing 'cf'.
Predict if the stock price will increase tomorrow by typing 'pp'.
Exit the program by typing 'x'.

For the prediction part, the model is trained using 75% of the available data and tested on the remaining 25%. The precision of the model on the test data is then displayed, followed by the prediction for tomorrow.

Note:
Ensure you have 'constituents_csv.csv' with the list of valid stock tickers for the program to run successfully.