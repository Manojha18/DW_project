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

# Exploring the data set
# TOP 5 ZIP CODES
# Taking the zip code and counting it using "value_counts" and displaying the top 10
top_zip_5 = pd.DataFrame(data['zip'].value_counts().head(5))
# Reset_index is used when you don't want to save it as a column
top_zip_5.reset_index(inplace=True)
top_zip_5.columns = ['ZIP', 'Total_Count']
print(top_zip_5)

# TOP 5 CITIES FOR EMERGENCY CALLS
top_twp_5 = pd.DataFrame(data['twp'].value_counts().head(5))
top_twp_5.reset_index(inplace=True)
top_twp_5.columns=['TWP', 'Total_count']
print(top_twp_5.head())

# VISUALIZATION
top_5_plot = plt.figure(figsize=(5,5))
sns.barplot(x='ZIP', y='Total_Count', data=top_zip_5)
top_5_plot.tight_layout()
plt.show()

top_twp_plot = plt.figure(figsize=(5,5))
label_twp=sns.barplot(x='TWP', y='Total_count', data= top_twp_5, palette="viridis")
label_twp.set_xticklabels(label_twp.get_xticklabels(),rotation=90)
plt.show()





