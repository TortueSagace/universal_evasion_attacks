import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import time
import re
import random
import copy

# For tqdm, detect if we are in a notebook environment or not.
try:
    from IPython import get_ipython
    isnotebook = get_ipython() is not None
except:
    isnotebook = False
if isnotebook:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

def L2_norm(x, x_initial):
    return np.linalg.norm(x - x_initial)

# Import the Iris dataset and the necessay functions from Scikit-learn.
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score

# Import MNIST
# from sklearn.datasets import fetch_openml

# Import XGBClassifier from XGBoost, LightGBMClassifier from LightGBM, HistGradientBoostingClassifier from Scikit-learn and ExtraTreesClassifier from Scikit-learn.
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
