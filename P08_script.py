# import libraires
import pandas as pd
import numpy as np
from finta import TA

# Loading dataset
df = pd.read_csv("/content/drive/My Drive/Ingénieur ML - OC/P8/datasets/train.csv")
df_sup = pd.read_csv("/content/drive/My Drive/Ingénieur ML - OC/P8/datasets/supplemental_train.csv")
asset = pd.read_csv("/content/drive/My Drive/Ingénieur ML - OC/P8/datasets/asset_details.csv")
print(df.head())