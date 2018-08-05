import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

data = pd.read_csv('E:/Summer 18/Data management warehousing and analytics/Project/911.csv')
print(data.head())
print(data.info())

# Removing unnecessary columns
data = data.drop('e', axis=1)
print(data.head(2))

