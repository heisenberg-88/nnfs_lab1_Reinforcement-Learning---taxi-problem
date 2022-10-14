import numpy as np
import pandas as pd
import os
from IPython.display import display


def viewdata():
    data = np.load("data.npy")
    print(data.shape)
    df = pd.DataFrame(data)
    display(df)

    if os.path.exists("data.csv"):
        os.remove("data.csv")
    df.to_csv('data.csv')