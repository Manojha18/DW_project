import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

data = pd.read_csv('E:/Summer 18/Data management warehousing and analytics/Project/911.csv')
print(data.head())
print(data.info())

# Removing unnecessary columns
data = data.drop('e', axis=1)
print(data.head(2))

# Exploring the data set
# TOP 5 ZIP CODES
# Taking the zip code and counting it using "value_counts" and displaying the top 10
top_zip_5 = pd.DataFrame(data['zip'].value_counts().head(5))
# Reset_index is used when you don't want to save it as a column
top_zip_5.reset_index(inplace=True)
top_zip_5.columns = ['ZIP', 'Total_Count']
print(top_zip_5)

