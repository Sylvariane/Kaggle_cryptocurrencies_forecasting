################################
########## LIBRAIRIES ###########
################################

import numpy as np
import dask.dataframe as dd
import matplotlib.pyplot as plt


################################
########## FUNCTIONS ###########
################################

def load_dataframe(filename):
    df = dd.read_csv(filename)
    return df

def feature_engineering(dataset):
    dataset["asset_ID"] = dataset["asset_ID"].map(dict_asset)
    cols = ['open', 'close', "high", "low"]
    for i in cols:
        dataset["log_"+i] = dataset[i].apply(lambda x: np.log(x))
    dataset["H-L"] = dataset["log_high"] - dataset["log_low"]
    dataset["O-C"] = dataset["log_open"] - dataset['log_close']
    dataset["MA_7min."] = dataset["log_close"].rolling(7).mean()
    dataset["MA_14min."] = dataset["log_close"].rolling(14).mean()
    dataset["MA_21min."] = dataset["log_close"].rolling(21).mean()
    dataset["STD_7min."] = dataset["log_close"].rolling(7).std()
    dataset["MA_7d"] = dataset["log_close"].rolling(10080).mean()
    dataset["MA_14d"] = dataset["log_close"].rolling(20160).mean()
    dataset["MA_21d"] = dataset["log_close"].rolling(30240).mean()
    dataset["STD_7min."] = dataset["log_close"].rolling(10080).mean()
    return dataset

def cleaning_df(dataset):
    dataset = dataset.drop(["timestamp", "asset_ID", "count",
                  "open", "high", "low", "close",
                  "volume", "VWAP", "log_open",
                  "log_close", "log_high", "log_low"],
                  axis=1)
    dataset = dataset.fillna(0)
    dataset = dataset.dropna()
    return dataset

dict_asset = {
    0 : "Binance Coin",
    1 : "Bitcoin",
    2 : "Bitcoin Cash",
    3 : "Cardano",
    4 : "Dogecoin",
    5 : "EOS.IO",
    6 : "Ethereum",
    7 : "Ethereum Classic",
    8 : "IOTA",
    9 : "Litecoin",
    10 : "Maker",
    11 : "Monero",
    12 : "Stellar",
    13 : "TRON"
}

####################################
#### IMPORTING & CLEANSING DATA ####
####################################

df_crypto = load_dataframe('Datasets/train_data/train.csv')
df_crypto.columns = ["timestamp", "asset_ID", "count",
                     "open", "high", "low", "close",
                     "volume", "VWAP", "target"]
df_crypto = feature_engineering(df_crypto)
df_crypto = cleaning_df(df_crypto)

print(df_crypto.head())