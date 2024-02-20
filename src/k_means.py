import sys
import pandas as pd
import numpy as np
import random as rd

def read_data(data):
    df = pd.read_csv(data)
    return df.head()

if __name__ == '__main__':
    data = sys.argv[1]
    print(read_data(data))