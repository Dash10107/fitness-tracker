import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans
# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/02_data_outliers_removed_chauvenet.pkl")

predictor_columns = list(df.columns[:6])
 
 #Plot Settings
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20,5)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['lines.linewidth'] = 2

# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------

df.info()

for col in predictor_columns:
    df[col] = df[col].interpolate()
    
df.info()

# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------

duration = df[df['set']==1].index[-1]-df[df['set']==1].index[0]
duration.seconds

for s in df['set'].unique():
    start = df[df['set']==s].index[0]
    stop = df[df['set']==s].index[-1]
    duration = stop-start
    df.loc[(df['set']==s),'duration'] = duration.seconds

duration_df = df.groupby('category')['duration'].mean()    
duration_df[0]/5
duration_df[1]/10
# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------

df_lowpass = df.copy()
LowPass = LowPassFilter()
fs = 1000/200
cutoff = 1.3
df_lowpass = LowPass.low_pass_filter(
    df_lowpass,"acc_y",fs, cutoff, order=5
)


subset = df_lowpass[df_lowpass['set']==45]
fig,ax = plt.subplots(nrows=2,sharex=True,figsize=(20,10))
ax[0].plot(subset['acc_y'].reset_index(drop=True),label='Original')
ax[1].plot(subset['acc_y_lowpass'].reset_index(drop=True),label='Lowpass')

ax[0].legend(loc='upper center',bbox_to_anchor=(0.5, 1.15),shadow=True, fancybox=True)
ax[1].legend(loc='upper center',bbox_to_anchor=(0.5, 1.15),shadow=True, fancybox=True)

for col in predictor_columns:
    df_lowpass = LowPass.low_pass_filter(
        df_lowpass,col,fs, cutoff, order=5
    )
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    del df_lowpass[col + "_lowpass"]

# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------
df_pca = df_lowpass.copy()
PCA = PrincipalComponentAnalysis()

pc_values =  PCA.determine_pc_explained_variance(df_pca,predictor_columns)

plt.figure(figsize=(10,10))
plt.plot(range(1,len(predictor_columns)+1),pc_values)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance')
plt.title('Scree Plot')
plt.show()

df_pca = PCA.apply_pca(df_pca,predictor_columns,3)


subset = df_pca[df_pca['set']==35]
subset[['pca_1','pca_2','pca_3']].plot(figsize=(20,5))

# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------
df_squared = df_pca.copy()
acc_r = df_squared['acc_x']**2 + df_squared['acc_y']**2 + df_squared['acc_z']**2
gyr_r = df_squared['gyr_x']**2 + df_squared['gyr_y']**2 + df_squared['gyr_z']**2

df_squared['acc_r'] = np.sqrt(acc_r)
df_squared['gyr_r'] = np.sqrt(gyr_r)

subset = df_squared[df_squared['set']==14]
subset[['acc_r','gyr_r']].plot(subplots=True,figsize=(20,10))


# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------

df_temporal = df_squared.copy()
NumericalAbstraction = NumericalAbstraction()

predictor_columns = predictor_columns + ['acc_r','gyr_r']

ws = int(1000/200)

for col in predictor_columns:
    df_temporal = NumericalAbstraction.abstract_numerical(df_temporal,[col],ws,"mean")
    df_temporal = NumericalAbstraction.abstract_numerical(df_temporal,[col],ws,"std")

df_temporal_list = []
for s in df_temporal['set'].unique():
    subset = df_temporal[df_temporal['set']==s].copy()
    for col in predictor_columns:
        subset = NumericalAbstraction.abstract_numerical(subset,[col],ws,"mean")
        subset = NumericalAbstraction.abstract_numerical(subset,[col],ws,"std")
    df_temporal_list.append(subset)
    
df_temporal = pd.concat(df_temporal_list)

df_temporal.info()

subset[['acc_x','acc_x_temp_mean_ws_5','acc_x_temp_std_ws_5']].plot()
# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------

df_frequency = df_temporal.copy().reset_index()
FreqAbs = FourierTransformation()

fs = int(1000 / 200)
ws = int(2800 / 200)

# df_frequency = FreqAbs.abstract_frequency(df_frequency,["acc_x"],ws,fs)
# subset = df_frequency[df_frequency['set']==15]
# df_frequency.info()
# subset[['acc_x','acc_x_freq_2.143_Hz_ws_14','acc_x_freq_2.5_Hz_ws_14']].plot(figsize=(20,5))

df_frequency_list = []
for s in df_frequency['set'].unique():
    subset = df_frequency[df_frequency['set']==s].reset_index(drop=True).copy()
    subset = FreqAbs.abstract_frequency(subset,predictor_columns,ws,fs)
    df_frequency_list.append(subset)

df_frequency = pd.concat(df_frequency_list).set_index('epoch (ms)',drop=True)

# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------

df_frequency = df_frequency.dropna()
df_frequency = df_frequency.iloc[::2]

# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------
df_cluster = df_frequency.copy()
cluster_columns = ['acc_x','acc_y','acc_z']
k_values = range(2,10)
inertia = []

for k in k_values:
    subset = df_cluster[cluster_columns]
    kmeans = KMeans(n_clusters=k,n_init=20, random_state=0)
    cluster_labels = kmeans.fit_predict(subset)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(20,10))
plt.plot(k_values,inertia)
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()


kmeans = KMeans(n_clusters=5,n_init=20, random_state=0)
subset = df_cluster[cluster_columns]
df_cluster['cluster'] =  kmeans.fit_predict(subset)

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot( projection='3d')
for c in df_cluster['cluster'].unique():
    subset = df_cluster[df_cluster['cluster']==c]
    ax.scatter(subset['acc_x'],subset['acc_y'],subset['acc_z'],label='Cluster '+str(c))
ax.set_xlabel('acc_x')
ax.set_ylabel('acc_y')
ax.set_zlabel('acc_z')
plt.legend()
plt.show()

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot( projection='3d')
for l in df_cluster['label'].unique():
    subset = df_cluster[df_cluster['label']==l]
    ax.scatter(subset['acc_x'],subset['acc_y'],subset['acc_z'],label='Label '+str(l))
ax.set_xlabel('acc_x')
ax.set_ylabel('acc_y')
ax.set_zlabel('acc_z')
plt.legend()
plt.show()


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

df_cluster.to_pickle("../../data/interim/03_data_features.pkl")