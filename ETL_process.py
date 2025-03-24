
import os
from dotenv import load_dotenv
import pandas as pd
import simfin as sf
from simfin.names import *
import numpy as np

# get the API token from the environment variables
load_dotenv()
api_key = os.getenv('API_KEY')

#create a directory
os.makedirs("data/RAW", exist_ok=True)


# Extracting & Saving Raw Data

# Loads daily stock prices, annual income statements and metadata about companies using the SimFin API.
# Saves data to stock_prices.csv, us_income_statements.csv and us_companies_list.csv.


# Defining the functions to download bulk data from the API
def load_price_data():
    """
    Loads daily stock prices for all US companies from SimFin 
    and saves the raw data to a specified directory for processing.
    """
    sf.set_api_key(api_key)
    sf.set_data_dir("~/simfin_data/")

    # Load daily stock prices for all US companies
    print("📥 Downloading US stock market data...")
    df_prices = sf.load_shareprices(market="us", variant="daily")

    # Reset index to make 'Date' a normal column
    df_prices = df_prices.reset_index()

    # Define save path - ADAPT TO YOUR LOCAL ENVIRONMENT
    save_path = "data/RAW/stock_prices.csv"

    # Save raw data to CSV
    df_prices.to_csv(save_path, index=False)

    print(f"✅ Data saved to: {save_path}")

def load_income_statements():
    """
    Loads income statements for all US companies from SimFin 
    and saves the raw data to a specified directory for processing.
    """
    # Download all US Income Statements
    df_income = sf.load_income(variant='annual', market='us')
    save_path = "data/RAW/us_income_statements.csv"
    df_income.to_csv(save_path, index=False)
    print(f"✅ Data saved to: {save_path}")

def load_companies_data():
    """
    Loads company information for all US companies from SimFin 
    and saves the raw data to a specified directory for processing.
    """
    # Download all US Income Statements
    df_companies = sf.load_companies(market='us')
    save_path = "data/RAW/us_companies_list.csv"
    df_companies.to_csv(save_path, index=False)
    print(f"✅ Data saved to: {save_path}")
  
    
 
def merge_and_preprocess_data(df_prices, df_income):
    """
    Preprocessing & Saving Preprocessed Data

    Loads from the RAW layer stock_prices.csv and us_income_statements.csv, and merges them using SimFinId as the key.
    Then, it cleans and prepares the data:
    - Drops the "Dividend" column (mostly null).
    - Converts "Date" to datetime.
    - Handles nulls in "Shares Outstanding" intelligently using group-wise .last() and .map().
    - Drops or fills missing values using thresholds, medians (for numeric), and modes (for categorical).
    - Applies linear interpolation for remaining NaNs.
    - Saves the cleaned version in the PREPROC folder.   
    """

    # Step 1: Convert dates to datetime
    df_prices['Date'] = pd.to_datetime(df_prices['Date'])
    df_income['Publish Date'] = pd.to_datetime(df_income['Publish Date'])

    # Step 2: Sort BOTH DataFrames by SimFinId and their respective date fields
    df_prices_sorted = df_prices.sort_values(by=['SimFinId', 'Date']).reset_index(drop=True)
    df_income_sorted = df_income.sort_values(by=['SimFinId', 'Publish Date']).reset_index(drop=True)

    # Make sure both date columns are datetime type
    df_prices_sorted['Date'] = pd.to_datetime(df_prices_sorted['Date'])
    df_income_sorted['Publish Date'] = pd.to_datetime(df_income_sorted['Publish Date'])

    # Sort both DataFrames by the merge keys
    df_prices_sorted = df_prices_sorted.sort_values(by=['SimFinId', 'Date'])
    df_income_sorted = df_income_sorted.sort_values(by=['SimFinId', 'Publish Date'])

    # Perform the merge using groupby and apply
    def merge_asof_group(group):
        return pd.merge_asof(
            group[0],
            group[1],
            left_on='Date',
            right_on='Publish Date',
            direction='backward'
        )

    # Group by 'SimFinId' and apply the merge_asof function
    df_merged = df_prices_sorted.groupby('SimFinId').apply(
        lambda x: merge_asof_group((x, df_income_sorted[df_income_sorted['SimFinId'] == x.name]))
    ).reset_index(drop=True)


    # Check if there are any mismatches in the new merged DataFrame
    df_merged['SimFinId_match'] = df_merged['SimFinId_x'] == df_merged['SimFinId_y']
    mismatch_rows_correct = df_merged[df_merged['SimFinId_match'] == False]
    # Display the mismatch rows in the new merged DataFrame
    mismatch_rows_correct[['Ticker','Date','Close','Revenue','SimFinId_x','SimFinId_y', 'SimFinId_match']]

    df_merged.rename(columns={'SimFinId_x':'SimFinId'}, inplace=True)
    df_merged.drop(columns=['SimFinId_y','SimFinId_match'], inplace=True)

    df_merged.describe()

    # Check for null values
    df_merged.isna().mean() # 99% of the dividend column is null for all tickers 

    # Changing Date into a Datetime Object
    df_merged['Date'] = pd.to_datetime(df_merged['Date']) # converting the "Date" column to a Datetime Object 

    # Checking for Duplicates
    df_merged.duplicated().sum()  # Count duplicates

    # Dropping the "Divident" column with 99% of nulls 
    df_merged = df_merged.drop(columns=["Dividend"])

    # Handling the null values in "Shares Outstanding" 
    # Sort the DataFrame by Ticker and Date in ascending order
    df_merged = df_merged.sort_values(by=["Ticker", "Date"])

    # Get the latest (most recent) non-null "Shares Outstanding" per Ticker
    latest_shares_outstanding = df_merged.groupby("Ticker")["Shares Outstanding"].last()

    # Fill missing values using the latest reported value per Ticker
    df_merged["Shares Outstanding"] = df_merged["Shares Outstanding"].fillna(df_merged["Ticker"].map(latest_shares_outstanding))

    # The reason some “Shares Outstanding” values are still null, even after replacing them with the latest available value, is that .last() in groupby("Ticker") selects only the most recent non-null value per Ticker. However, if a Ticker never had a reported non-null value, .last() returns NaN instead of a valid number. As a result, when we try to map and fill missing values, there is no valid value available for replacement, leaving some entries still null.
    null_so =  df_merged[df_merged["Shares Outstanding"].isna()]

    # Replacing the Shares Outstanding Values that have never had a reported non-null vallue with 0. 
    df_merged["Shares Outstanding"] = df_merged["Shares Outstanding"].fillna(0) # replacing the remaining "Shares Outstanding" with 0


    # Drop Columns with Too Many Missing Values (>70%)
    # If a column has too many missing values and isn't critical, it's best to drop it.
    # columns dropped: Research & Development, Depreciation & Amortization
    threshold = 0.7  # 70% missing values threshold
    df_merged = df_merged.dropna(axis=1, thresh=len(df_merged) * (1 - threshold))

    # Fill the missing values with the next available or latest income information for each company. By applying both ffill and bfill on the entire DataFrame within each SimFinId group, we ensure that missing values are filled with the closest available data, whether it comes from a previous or a future date. This should help fill in the missing values more effectively.
    # Sort the DataFrame by SimFinId and Date
    df_merged = df_merged.sort_values(by=['SimFinId', 'Date'])

    # Forward fill and backward fill missing values within each SimFinId group
    df_merged = df_merged.groupby('SimFinId').apply(lambda group: group.ffill().bfill())

    # Reset index after groupby
    df_merged = df_merged.reset_index(drop=True)

    # Fill Categorical Columns with Mode
    # For categorical columns like Currency, Ticker, or Fiscal Period.
    # - Categorical columns like Currency, Ticker, and Fiscal Period contain discrete values.
    # - The mode is the most frequently occurring category, making it a logical replacement for missing values.
    # - If we later use one-hot encoding or label encoding, missing values could create problems.
    # - Filling with the mode ensures that all rows have valid categorical values.

    # Fill missing values for categorical columns
    cat_cols = ['Currency', 'Ticker']
    for col in cat_cols:
        df_merged[col] = df_merged.groupby('SimFinId')[col].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else ''))

    # Fill missing values for Fiscal Period using the year from Publish Date
    df_merged['Fiscal Period'] = df_merged['Fiscal Period'].fillna(df_merged['Publish Date'].dt.year.astype(str))

    # Drop Rows with Too Many Nulls
    # If some rows still have excessive missing values, drop them.
    df_merged.dropna(thresh=len(df_merged.columns) * 0.7, inplace=True)  # Drops rows with >30% missing


    # Interpolation for Time-Series Data
    # Uses linear interpolation to estimate missing values between known values.
    # Works by computing a straight-line equation between two known points and filling missing values accordingly.
    # - If your dataset follows a time pattern (e.g., financial data), use interpolation.
    # - For time-series financial data (e.g., stock prices, revenue), missing values often occur due to holidays, reporting delays, or system gaps.
    # - Interpolation helps smooth data trends rather than using arbitrary imputation like mean/median.
    # - It ensures that missing values follow the trend instead of creating unnatural spikes.
    df_merged = df_merged.sort_values('Date')  # Ensure sorting before interpolation
    df_merged.interpolate(method='linear', inplace=True)

    # Calculate the percentage of missing values for each column
    missing_values = df_merged.isna().mean()
    # Filter columns with missing values greater than 0
    missing_columns = missing_values[missing_values > 0]

    # Fill missing values with 0 for the columns with missing values
    df_merged[missing_columns.index] = df_merged[missing_columns.index].fillna(0)

    # Save merged and preprocessed dataset
    os.makedirs("data/PREPROCESSING", exist_ok=True)
    df_merged.to_csv("data/PREPROCESSING/merged_stock_income.csv", index=False) 
    return df_merged   
    
    
def enrich_data(df_merged):
    """
    FEATURE ENGINEERING
    This notebook logically follows the preprocessing step and performs feature engineering for modeling or further analysis.
    
    Loads the cleaned data from the previous step
    Creates new features: 
    # Price Change % → Added daily price change % calculation (1 day lag)
    # Weekday (Weekday) → Categorical feature (Monday-Sunday) for classification models.
    # Month (Month) → Categorical month number (1-12) for classification models.
    # Quarter (Quarter) → Categorical feature (1-4) for classification models.
    # Daily Returns (Daily_Return) → Daily differences in stock prices
    # Moving Averages (MA_5, MA_10) → 5-day and 10-day moving averages for trend detection.
    # Log Returns (Log_Returns) → Helps in normalizing price data for time series models.
    """
    #Creating Price Change %
    df_merged["Price Change %"] = df_merged["Close"].pct_change() * 100  # Use correct column name

    # Handling Null Values: Since .pct_change() creates NaN values for the first row, fill them:
    df_merged["Price Change %"].fillna(0, inplace=True)
    df_merged["Target"].fillna(0, inplace=True)
    df_merged.head()

    # Weekday column
    df_merged['Date'] = pd.to_datetime(df_merged['Date']) # converting the "Date" column to a Datetime Object (double checking)
    df_merged['Weekday'] = df_merged['Date'].dt.day_name()  # 'Monday', 'Tuesday', etc.

    # Month Column
    df_merged['Month'] = df_merged['Date'].dt.month  # 1 (Jan) to 12 (Dec)

    # Quarter Column
    df_merged['Quarter'] = df_merged['Date'].dt.quarter  # 1 to 4

    # Daily Returns (Percentage Change)
    df_merged['Daily_Return'] = df_merged['Close'] / df_merged['Close'].shift(1)

    # Moving Averages (5-day and 10-day)
    df_merged['MA_5'] = df_merged['Close'].rolling(window=5, min_periods=1).mean()
    df_merged['MA_10'] = df_merged['Close'].rolling(window=10, min_periods=1).mean()


    # Daily Log Returns for Time Series Modeling
    df_merged.set_index('Date', inplace=True)
    df_merged['Log_Returns'] = np.log(df_merged['Close'] / df_merged['Close'].shift(1))
    df_merged['Log_Returns'].dropna(inplace=True)
    # Replace inf/-inf with NaN
    df_merged['Log_Returns'].replace([np.inf, -np.inf], np.nan, inplace=True)
    # Drop missing values
    df_merged.dropna(subset=['Log_Returns'], inplace=True)

    # Save enriched dataset
    os.makedirs("data/ENRICH", exist_ok=True)
    df_merged.to_csv("data/ENRICH/merged_stock_income.csv", index=True) 
    return df_merged  
    
    
    
    
    

if __name__ == "__main__":
    # RAW - DOWNLOAD BULK DATA FROM API AND SAVE INTO RAW FOLDER
    load_price_data()
    load_income_statements()
    load_companies_data()
    
    # PREPROCESSING - LOAD DATA FROM RAW LAYER, MERGE AND PREPROCESS AND SAVE INTO PREPROCESSING FOLDER
    # Load the stock prices dataset
    df_prices = pd.read_csv("data\RAW\stock_prices.csv") 
    # Load the income statements dataset
    df_income = pd.read_csv("data/RAW/us_income_statements.csv")
    #Run the merge and preprocessing 
    merge_and_preprocess_data(df_prices, df_income)
    
    # ENRICH - LOAD DATA FROM PREPROCESSING LAYER, CREATE NEW FEATURES AND SAVE INTO ENRICH FOLDER
    file_path = "data\PREPROCESSING\merged_stock_income.csv" 
    df_merged = pd.read_csv(file_path)
    enrich_data(df_merged)
    