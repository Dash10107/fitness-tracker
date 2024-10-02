import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter
from scipy.signal import argrelextrema
from sklearn.metrics import mean_absolute_error

pd.options.mode.chained_assignment = None


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/01_data_processed.pkl")
df = df[df["label"] != "rest"]

acc_r = df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2
gyr_r = df['gyr_x']**2 + df['gyr_y']**2 + df['gyr_z']**2
df['acc_r'] = np.sqrt(acc_r)
df['gyr_r'] = np.sqrt(gyr_r)


# --------------------------------------------------------------
# Split data
# --------------------------------------------------------------
bench_df = df[df["label"] == "bench"]
squat_df = df[df["label"] == "squat"]
row_df = df[df["label"] == "row"]
ohp_df = df[df["label"] == "ohp"]
deadlift_df = df[df["label"] == "dead"]



# --------------------------------------------------------------
# Visualize data to identify patterns
# --------------------------------------------------------------

plot_df = bench_df
plot_df[plot_df['set']==plot_df['set'].unique()[0]]['acc_x'].plot()
plot_df[plot_df['set']==plot_df['set'].unique()[0]]['acc_y'].plot()
plot_df[plot_df['set']==plot_df['set'].unique()[0]]['acc_z'].plot()
plot_df[plot_df['set']==plot_df['set'].unique()[0]]['acc_r'].plot()

plot_df[plot_df['set']==plot_df['set'].unique()[0]]['gyr_x'].plot()
plot_df[plot_df['set']==plot_df['set'].unique()[0]]['gyr_y'].plot()
plot_df[plot_df['set']==plot_df['set'].unique()[0]]['gyr_z'].plot()
plot_df[plot_df['set']==plot_df['set'].unique()[0]]['gyr_r'].plot()


# --------------------------------------------------------------
# Configure LowPassFilter
# --------------------------------------------------------------

fs = 1000/200

LowPassFilter = LowPassFilter()

# --------------------------------------------------------------
# Apply and tweak LowPassFilter
# --------------------------------------------------------------

bench_set = bench_df[bench_df['set']==bench_df['set'].unique()[0]]
sqaut_set = squat_df[squat_df['set']==squat_df['set'].unique()[0]]
row_set = row_df[row_df['set']==row_df['set'].unique()[0]]
ohp_set = ohp_df[ohp_df['set']==ohp_df['set'].unique()[0]]
deadlift_set = deadlift_df[deadlift_df['set']==deadlift_df['set'].unique()[0]]

columns = 'acc_y'
LowPassFilter.low_pass_filter(bench_set,columns, fs, 0.4,10)[columns + '_lowpass'].plot()
 # --------------------------------------------------------------
# Create function to count repetitions
# --------------------------------------------------------------
def count_reps(dataset,cutoff=0.4,order=10,column='acc_r'):    
    data = LowPassFilter.low_pass_filter(
        dataset,col=column,sampling_frequency= fs,cutoff_frequency= cutoff,order=order)
    indexes = argrelextrema(data[column + "_lowpass"].values,np.greater)
    peaks = data.iloc[indexes]
    
    # fig,ax = plt.subplots()
    # plt.plot(dataset[f'{column}_lowpass'])
    # plt.plot(peaks[f'{column}_lowpass'], 'o',color='red')
    # ax.set_ylabel(f'{column}_lowpass')
    # exercise = dataset['label'].iloc[0].title()
    # category = dataset['category'].iloc[0].title()
    # plt.title(f'{category}  {exercise} {len(peaks)}Repetitions')
    # plt.show()
    return len(peaks)

count_reps(bench_set,cutoff=0.4)
count_reps(sqaut_set,cutoff=0.35)
count_reps(row_set,cutoff=0.6,column='gyr_x')
count_reps(ohp_set,cutoff=0.35)
count_reps(deadlift_set,cutoff=0.4)



# --------------------------------------------------------------
# Create benchmark dataframe
# --------------------------------------------------------------

df['reps'] = df['category'].apply(lambda x: 5 if x == 'heavy' else 10)

rep_df = df.groupby(['category','label','set'])['reps'].max().reset_index()
rep_df['reps_pred']=0

for s in df['set'].unique():
    subset = df[df['set']==s]
    column = 'acc_r'
    cutoff = 0.4
    if subset['label'].iloc[0] == 'row':
        column = 'gyr_x'
        cutoff = 0.65
    if subset['label'].iloc[0] == 'squat':
        cutoff = 0.35
    if subset['label'].iloc[0] == 'ohp':
        cutoff = 0.35
    reps = count_reps(subset,cutoff=cutoff,column=column)   
    rep_df.loc[(rep_df['set']==s),'reps_pred'] = reps



# --------------------------------------------------------------
# Evaluate the results
# --------------------------------------------------------------

error = mean_absolute_error(rep_df['reps'],rep_df['reps_pred']).round(2)

rep_df.groupby(['category','label'])[['reps','reps_pred']].mean().plot.bar()