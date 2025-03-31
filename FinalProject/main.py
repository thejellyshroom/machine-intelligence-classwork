'''
Jessica Sheng
ACAD 222, Spring 2025
jlsheng@usc.edu
Final Project Part 1
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

'''datasets:
https://huggingface.co/datasets/ajibawa-2023/Children-Stories-Collection/blob/main/Children-Stories-0-Final.json
https://huggingface.co/datasets/ajibawa-2023/Children-Stories-Collection/blob/main/Children-Stories-1-Final.json
https://huggingface.co/datasets/ajibawa-2023/Children-Stories-Collection/blob/main/Children-Stories-2-Final.json
https://huggingface.co/datasets/ajibawa-2023/Children-Stories-Collection/blob/main/Children-Stories-3-Final.json
https://huggingface.co/datasets/ajibawa-2023/Children-Stories-Collection/blob/main/Children-Stories-4-Final.json
https://huggingface.co/datasets/ajibawa-2023/Children-Stories-Collection/blob/main/Children-Stories-5-Final.json
https://huggingface.co/datasets/ajibawa-2023/Children-Stories-Collection/blob/main/Children-Stories-6-Final.json
https://huggingface.co/datasets/ajibawa-2023/Children-Stories-Collection/blob/main/Children-Stories-7-Final.json
https://huggingface.co/datasets/ajibawa-2023/Children-Stories-Collection/blob/main/Children-Stories-8-Final.json
https://huggingface.co/datasets/ajibawa-2023/Children-Stories-Collection/blob/main/Children-Stories-9-Final.json
'''

# Load the dataset from huggingface
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'Children-Stories-0-Final.json')
df = pd.read_json(file_path)

#make sure data is properly formatted and there are no missing values
df.isnull().sum()
if df.isnull().sum().any():
    print("There are missing values in the dataset")
    df = df.dropna()
else:
    print("There are no missing values in the dataset")

# Print the first few rows of the dataset
print(df.head())

print(df.columns)
print(df.info())
print(df.describe())

# Print the correlation for numeric columns only
print("Correlation for numeric columns:")
numeric_cols = df.select_dtypes(include=['number']).columns
if len(numeric_cols) > 1:
    print(df[numeric_cols].corr())
else:
    print("There is only one numeric column ('text_token_length'), so correlation can't be calculated.")
