import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from matplotlib.colors import ListedColormap


def prepare_data(df):
  X = df.drop("y", axis=1)
  y = df["y"]
  return X,y