#!/usr/bin/python3
"""
Created on Thur Dec 28 03:20:25 2023

@author: Moronfoluwa Akintola
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def convert_csv(filename):
    """ 
        Read the CSV file, convert to a dataframe and return the same dataframe 
    """
    df = pd.read_csv(filename)
    return df

def read_worldbank_data(df):
    """ 

    """

    return df

if __name__ == "__main__":
    
    # Load data into pandas dataframe from CSV file
    df = convert_csv('./data.csv')
    df.head()

    # Your program should
    data = read_worldbank_data(df)
