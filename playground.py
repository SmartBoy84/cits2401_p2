import pandas as pd

df = pd.read_csv("data/weather_data.csv")
summary_map = {v: i + 1 for i, v in enumerate(list(set(df["Summary"])))}
import numpy as np

x = np.array([1,2,3])
print(x)