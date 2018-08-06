import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
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
# plt.show()

top_twp_plot = plt.figure(figsize=(5,5))
label_twp = sns.barplot(x='TWP', y='Total_count', data=top_twp_5, palette="viridis")
label_twp.set_xticklabels(label_twp.get_xticklabels(), rotation=90)
# plt.show()

# EXTRACTING THE FEATURES FROM THE COLUMNS
data['label'] = data['title'].str.split(':').str[0] #own
data['label']
print(data.head(2))

print(data['label'].nunique())
data['STA'] = data['desc'].str.split(';').str[2] #own
print(data['STA'].head(2))

print(data['label'].value_counts())
# Splitting hour,days of week, month and date from timeStamp
data['Hour'] = data.timeStamp.map(lambda x: pd.to_datetime(x).hour) #own
data['Days'] = data.timeStamp.map(lambda x: pd.to_datetime(x).dayofweek) #own
data['Month'] = data.timeStamp.map(lambda x: pd.to_datetime(x).month) #own


# Converting day values to strings
days = {0: 'Monday',
        1: 'Tuesday',
        2: 'Wednesday',
        3: 'Thursday',
        4: 'Friday',
        5: 'Saturday',
        6: 'Sunday'}

data['Days'] = data['Days'].map(days)
print(data['Hour'].head(2))
print(data.head(3))


# GEOGRAPHICAL ANALYSIS
# Grouping the cluster with +/- 4.5* std deviation to avoid outliers
geo_ana = data[(np.abs(data["lat"]-data["lat"].mean())
                <= (4.5*data["lat"].std())) & (np.abs(data["lng"]-data["lng"].mean())
                                              <= (10*data["lng"].std()))]


# Finding average lat and long
pd.options.mode.chained_assignment=None
avg_lat=geo_ana['lat'].mean()
avg_lng=geo_ana['lng'].mean()
geo_ana['x_lg']=geo_ana['lng'].map(lambda v: v-avg_lng)
geo_ana['y_lt']=geo_ana['lat'].map(lambda v: v-avg_lat)


X=geo_ana[['x_lg','y_lt']].reset_index().drop('index',axis=1)

# Kmeans Clustering

kmeans = KMeans(n_clusters=6)
kmeans.fit(X)
cluster_6=plt.figure(figsize=(5,5))
plt.scatter(X['x_lg'],X['y_lt'],c=kmeans.labels_,cmap='plasma')
plt.xlim(-0.3,0.3)
plt.show()

print(kmeans.cluster_centers_)

print(kmeans.labels_)

print(kmeans.inertia_)

# Finding the area of township
# A = 2.pi.R^2 |sin(lat1)-sin(lat2)| |lon1-lon2|/360

dis_lat = np.abs(np.sin(np.max(geo_ana["lat"])/180*np.pi)-np.sin(np.min(geo_ana["lat"])/180*np.pi))

dis_lng = np.abs(np.max(geo_ana["lng"])-np.min(geo_ana["lng"]))

# Radius of the earth is 6371


def area(lat_a,lng_b):

    return np.pi*(6371**2)*dis_lat*lng_b/180


Area_t=area(dis_lat,dis_lng)
Area_t = np.int(Area_t)
print("Area of the Township {} sq. km".format(Area_t))


population=np.int(Area_t*314)
print("Avg Population of the Township {}".format(population))


cluster=pd.concat([geo_ana.reset_index().drop('index',axis=1),pd.DataFrame(kmeans.labels_,columns=['C'])],axis=1)
cluster.head(2)

d_c=pd.concat([geo_ana.reset_index().drop('index',axis=1),pd.DataFrame(kmeans.labels_,columns=['C'])],axis=1)
Pop_density=d_c.groupby(['lat','lng']).count().reset_index().drop(['zip','twp','label','Hour','Days','Month','STA','C','x_lg','y_lt'],axis=1)

print(Pop_density.head(1))

A=Pop_density[['lat','lng']]
B=Pop_density.drop(['lat','lng'],axis=1)

# Population of each cluster

from sklearn.neighbors import KernelDensity
kd=KernelDensity()
kd.fit(A,B)
avg=np.exp(kd.score_samples(A)).mean()

mean_p=0.95*population/Area_t
B=pd.DataFrame(np.exp(kd.score_samples(A)))
min_v=np.exp(kd.score_samples(A)).min()


def kernel(ker_pop):

    return (((avg+np.sign(ker_pop-avg)*(np.abs(ker_pop-avg))**0.59)/avg))*mean_p

B=pd.DataFrame(B.apply(kernel))


pd.DataFrame(pd.concat([A,B],axis=1).head(5))

poly=PolynomialFeatures(2)
a_c=pd.DataFrame(poly.fit_transform(d_c[['lat','lng']]),columns=['1','lat','lng','lat^2','lat*lng','lng^2'])
X_quad=pd.DataFrame(poly.fit_transform(A),columns=['1','lat','lng','lat^2','lat*lng','lng^2'])


quad_model=LinearRegression()
quad_model.fit(X_quad,B)
popdense=pd.DataFrame(quad_model.predict(a_c).ravel(),columns=["Pop_D"])
data_clus=pd.concat([d_c,popdense],axis=1)
print(data_clus.head(2))

# Grouping by clusters
p_c=d_c.groupby('C').mean()['Pop. Density'].as_matrix()
print(p_c)
p_f=np.round(p_c*100)/100
print(p_f)
print("predicted population densities:\n")
print(p_f)

# Approximate cluster area and predicted population density
# Reference:"911 Calls - City Sevices Planning For Emergencies | Kaggle", Kaggle.com, 2018.
app_area=[]
print("The approximate cluster areas are:\n")
for a in range(0,6):
    tempdata=d_c[d_c['C']==a]
    l=np.abs(np.sin(np.max(tempdata["lat"])/180*np.pi)-np.sin(np.min(tempdata["lat"])/180*np.pi))
    lg=np.abs(np.max(tempdata["lng"])-np.min(tempdata["lng"]))
    p=(2/3)*area(l,lg)
    app_area.append(p)
    print("Cluster {} : {:.2f} sq km".format(a+1,p))
print("\Predicted Population:")
c_pop=(app_area*p_f)
c_pop1=np.round(c_pop*100)/100
print(c_pop1)



