import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

data = pd.read_csv('E:/Summer 18/Data management warehousing and analytics/Project/911.csv')
print(data.head())

top_10_zip = pd.DataFrame(data['zip'].value_counts().head(10))
top_10_zip.reset_index(inplace=True)
top_10_zip.columns = ['ZIP', 'Count']
print(top_10_zip)