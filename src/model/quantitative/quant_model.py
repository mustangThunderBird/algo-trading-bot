import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
import platform

# Suppress TensorFlow logs
import absl.logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
absl.logging.set_verbosity(absl.logging.ERROR)

if os.name == "posix":
    system_name = platform.system()
    if system_name == "Linux":
        os.system("export QT_QPA_PLATFORM_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms")

def preprocess_all_stocks_data(filepath:str) -> dict:
    '''
    Takes the all_stock_data.csv and returns a dictionary of data frames for each
    ticker in the data
    '''
    df = pd.read_csv(filepath)
    #start data from index 3 to skip header rows
    df = df[3:]
    #extact the date col and set it as the index
    date_col = df.iloc[:,0]
    df = df.set_index(date_col)
    df.index.name = 'Date'
    # get rid of the 0th column now
    df = df.iloc[:, 1:]

    #Identify the ticker group size
    group_size = 6
    num_cols = df.shape[1]
    num_stocks = num_cols // group_size

    #process each stock data
    reformatted_data = {}
    ticker_start = 0
    ticker_end = 6
    cols_names = ['Adj Close','Close','High','Low','Open','Volume']
    for i in range(num_stocks):
        temp = df.iloc[:,ticker_start:ticker_end]
        ticker_symbol = temp.columns[0]
        temp.Name = ticker_symbol
        temp.columns = cols_names
        if reformatted_data.get(ticker_symbol) == None:
            reformatted_data[ticker_symbol] = temp
        ticker_start += 6
        ticker_end += 6
    
    return reformatted_data

def compute_rsi(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_obv(close, volume):
    obv = [0]
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i - 1]:
            obv.append(obv[-1] + volume.iloc[i])
        elif close.iloc[i] < close.iloc[i - 1]:
            obv.append(obv[-1] - volume.iloc[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=close.index)

def compute_macd(series, short_period=12, long_period=26, signal_period=9):
    short_ema = series.ewm(span=short_period, adjust=False).mean()
    long_ema = series.ewm(span=long_period, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal_period, adjust=False).mean()
    return macd, signal_line
    
def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filename}")

def load_model(filename):
    if os.path.exists(filename):
        print(f"Loading model from {filename}")
        with open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        print(f"No model found at {filename}")
        return None

def preprocess_ticker_data(data: pd.DataFrame) -> pd.DataFrame:
    for col in data.columns:
        # Ensure 'Close' column is numeric
        data[col] = pd.to_numeric(data[col], errors='coerce')

        # Handle cases where  values are converted to NaN
        if data[col].isnull().any():
            print(f"Warning: Non-numeric data found in 'Close' column. {data['Close'].isnull().sum()} rows will be dropped.")
            data.dropna(subset=['Close'], inplace=True)

    data['Daily_Return'] = data['Close'].pct_change()

    for i in range(1, 5):
        data[f'Return_Lag{i}'] = data['Daily_Return'].shift(i)

    # Calculate additional return-based features
    data['ROC_5'] = data['Close'].pct_change(periods=5)
    data['MA_Return_5'] = data['Daily_Return'].rolling(window=5).mean()
    data['Volatility_5'] = data['Daily_Return'].rolling(window=5).std()
    data['Volatility_10'] = data['Daily_Return'].rolling(window=10).std().fillna(0)
    data['RSI'] = compute_rsi(data['Close'])
    data['OBV'] = compute_obv(data['Close'], data['Volume']).pct_change()
    data['MACD'], data['MACD_Signal'] = compute_macd(data['Close'])

    # Scale RSI to be between 0 and 1
    data['RSI'] = data['RSI'] / 100.0

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)
    return data

def build_quant_model(ticker:str, data:pd.DataFrame, force_rebuild=False) -> StackingRegressor:
    model_filename = os.path.join(os.path.dirname(__file__), 'models', f"{ticker}_quant_model.pkl")

    loaded_model = load_model(model_filename)
    if loaded_model and not force_rebuild:
        return loaded_model

    try:
        df_clean = preprocess_ticker_data(data)
        feature_columns = ['Return_Lag1', 'Return_Lag2', 'Return_Lag3',  'Return_Lag4',
           'ROC_5', 'MA_Return_5', 'Volatility_5', 'Volatility_10', 'RSI', 'OBV', 'MACD', 
           'MACD_Signal']

        #Seperate data
        X = df_clean[feature_columns]
        y = df_clean['Daily_Return']

        #Split data
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.2, random_state=42, shuffle=False)

        #Build the XGBoost Model
        # Define the parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300, 400],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 10],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'reg_alpha': [0, 0.01, 0.1, 1],
            'reg_lambda': [0.1, 1, 10]
        }

        # Initialize the model
        xgb_model = XGBRegressor(random_state=42)

        # Set up RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_grid,
            n_iter=50,  # Number of random combinations to try
            scoring='neg_mean_squared_error',
            cv=3,  # Cross-validation folds
            verbose=2,
            random_state=42,
            n_jobs=-1  # Use all available cores
        )

        # Fit the model
        random_search.fit(X_train, y_train)

        # Best parameters and model
        best_xgb_model = random_search.best_estimator_
        print(f"Best Parameters For XGBoost: {random_search.best_params_}")

        #Build the random forest model
        # Define the parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [None, 10, 20, 30, 40],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2'],
            'bootstrap': [True, False]
        }

        # Initialize the model
        rf = RandomForestRegressor(
            random_state=42)

        # Set up RandomizedSearchCV
        rf_random_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_grid,
            n_iter=50,  # Number of random combinations to try
            scoring='neg_mean_squared_error',
            cv=3,  # Cross-validation folds
            verbose=2,
            random_state=42,
            n_jobs=-1  # Use all available cores
        )

        # Fit the model
        rf_random_search.fit(X_train, y_train)

        # Print the best parameters
        print(f"Best Parameters: {rf_random_search.best_params_}")

        # Get the best model
        best_rf_model = rf_random_search.best_estimator_
        # Define a meta-model
        meta_model = LinearRegression()

        # Create the stacking regressor with the tuned RF model
        stacking_regressor = StackingRegressor(
            estimators=[
                ('xgb', best_xgb_model), 
                ('rf', best_rf_model)],
            final_estimator=meta_model
        )

        # Fit the stacking model
        stacking_regressor.fit(X_train, y_train)
        y_pred = stacking_regressor.predict(X_test)
        mse = root_mean_squared_error(y_test, y_pred)
        print(f'Stacking Model Root Mean Squared Error: {mse}')

         # Define thresholds for accuracy
        threshold_1 = 0.01
        accuracy_within_threshold_1 = (abs(y_test - y_pred) <= threshold_1).mean() * 100
        print(f"Percentage of predictions within ±{threshold_1}: {accuracy_within_threshold_1:.2f}%")

        threshold_2 = 0.05
        accuracy_within_threshold_2 = (abs(y_test - y_pred) <= threshold_2).mean() * 100
        print(f"Percentage of predictions within ±{threshold_2}: {accuracy_within_threshold_2:.2f}%")


        # Save plot
        plot_path = os.path.join(os.path.dirname(__file__), 'imgs', f'{ticker}.png')
        plt.figure(figsize=(14, 5))
        plt.plot(y_test.values, label='Actual Return', color='blue')
        plt.plot(y_pred, label='Stacking Predicted Return', color='orange')
        plt.title('Actual vs. Stacking Predicted Returns')
        plt.xlabel('Sample Index')
        plt.ylabel('Daily Return')
        plt.legend()
        plt.savefig(plot_path)

        # Write results to a Markdown file
        md_path = os.path.join(os.path.dirname(__file__), 'model_performance', f"{ticker}_training_results.md")
        with open(md_path, 'w+') as f:
            f.write(f"# Model Training Results for {ticker}\n\n")
            f.write(f"## Root Mean Squared Error (RMSE)\n")
            f.write(f"- **RMSE**: {mse:.4f}\n\n")
            f.write(f"## Prediction Accuracy Within Thresholds\n")
            f.write(f"- **Percentage within ±{threshold_1:.2f}**: {accuracy_within_threshold_1:.2f}%\n")
            f.write(f"- **Percentage within ±{threshold_2:.2f}**: {accuracy_within_threshold_2:.2f}%\n\n")
            f.write(f"## Performance Plot\n")
            f.write(f"![Performance Plot](../imgs/{ticker}.png)\n")

        #save the model
        print(f"Saving model for {ticker}. Type: {type(stacking_regressor)}")
        save_model(stacking_regressor, model_filename)
        return stacking_regressor
    except Exception as e:
        print(f"Failed to build quant model for {ticker}... {e}")
        return None