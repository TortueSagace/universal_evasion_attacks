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

# install pip given get-pip on windows 10
# python -m pip install --upgrade pip