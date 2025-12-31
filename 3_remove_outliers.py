import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from sklearn.neighbors import LocalOutlierFactor  # pip install scikit-learn
# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("data_resampled.pkl")
outlier_columns = list(df.columns[:6])

# --------------------------------------------------------------
# Plotting outliers
# --------------------------------------------------------------

plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100    

df[outlier_columns[:3] + ["label"]].boxplot(by="label", figsize=(20, 10), layout=(1,3))
df[outlier_columns[3:] + ["label"]].boxplot(by="label", figsize=(20, 10), layout=(1,3))
def plot_binary_outliers(dataset, col, outlier_col, reset_index=True):
    dataset = dataset.dropna(axis=0, subset=[col, outlier_col])
    dataset[outlier_col] = dataset[outlier_col].astype(bool)

    if reset_index:
        dataset = dataset.reset_index(drop=True)

    fig, ax = plt.subplots()
    ax.plot(dataset.index[~dataset[outlier_col]],
            dataset[col][~dataset[outlier_col]], "+")
    ax.plot(dataset.index[dataset[outlier_col]],
            dataset[col][dataset[outlier_col]], "r+")

    plt.xlabel("samples")
    plt.ylabel(col)
    plt.legend(["no outlier", "outlier"])
    plt.show()

def mark_outliers_iqr(dataset, col):
    dataset = dataset.copy()

    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    dataset[col + "_outlier"] = (
        (dataset[col] < lower) | (dataset[col] > upper)
    )
    return dataset

# --------------------------------------------------------------
# Interquartile range (distribution based)
# --------------------------------------------------------------
# Plot a single column

col = "acc_x"
dataset = mark_outliers_iqr(df,col)
plot_binary_outliers(dataset=dataset, col=col, outlier_col=col+"_outlier",reset_index=True)

# Loop over all columns
for col in outlier_columns:
    dataset = mark_outliers_iqr(df,col)
    plot_binary_outliers(dataset=dataset, col=col, outlier_col=col+"_outlier",reset_index=True)
    
# --------------------------------------------------------------
# Chauvenets criteron (distribution based)
# --------------------------------------------------------------

# Check for normal distribution
df[outlier_columns[:3] + ["label"]].plot.hist(by="label", figsize=(20, 20), layout=(3,3))
df[outlier_columns[3:] + ["label"]].plot.hist(by="label", figsize=(20, 20), layout=(3,3))

# Insert Chauvenet's function
def mark_outliers_chauvenet(dataset, col, C=2):
    dataset = dataset.copy()
    # Compute the mean and standard deviation.
    mean = dataset[col].mean()
    std = dataset[col].std()
    N = len(dataset.index)
    criterion = 1.0 / (C * N)

    # Consider the deviation for the data points.
    deviation = abs(dataset[col] - mean) / std

    # Express the upper and lower bounds.
    low = -deviation / math.sqrt(C)
    high = deviation / math.sqrt(C)
    prob = []
    mask = []

    # Pass all rows in the dataset.
    for i in range(0, len(dataset.index)):
        # Determine the probability of observing the point
        prob.append(
            1.0 - 0.5 * (scipy.special.erf(high[i]) - scipy.special.erf(low[i]))
        )
        # And mark as an outlier when the probability is below our criterion.
        mask.append(prob[i] < criterion)
    dataset[col + "_outlier"] = mask
    return dataset




# Loop over all columns

for col in outlier_columns:
    dataset = mark_outliers_chauvenet(df,col)
    plot_binary_outliers(dataset=dataset, col=col, outlier_col=col+"_outlier",reset_index=True)
# --------------------------------------------------------------
# Local outlier factor (distance based)
# --------------------------------------------------------------

# Insert LOF function
def mark_outliers_lof(dataset, columns, n=20):
    dataset = dataset.copy()

    lof = LocalOutlierFactor(n_neighbors=n)
    data = dataset[columns]
    outliers = lof.fit_predict(data)
    X_scores = lof.negative_outlier_factor_

    dataset["outlier_lof"] = outliers == -1
    return dataset, outliers, X_scores
# Loop over all columns

dataset, outliers, X_scores = mark_outliers_lof(df,outlier_columns)
for col in outlier_columns:
    plot_binary_outliers(dataset=dataset, col=col, outlier_col="outlier_lof",reset_index=True)

# --------------------------------------------------------------
# Check outliers grouped by label
# --------------------------------------------------------------

label="bench"
for col in outlier_columns:
    dataset = mark_outliers_iqr(df[df["label"]==label],col)
    plot_binary_outliers(dataset=dataset, col=col, outlier_col=col+"_outlier",reset_index=True)

label="bench"
for col in outlier_columns:
    dataset = mark_outliers_chauvenet(df[df["label"]==label],col)
    plot_binary_outliers(dataset=dataset, col=col, outlier_col=col+"_outlier",reset_index=True)
    
dataset, outliers, X_scores = mark_outliers_lof(df[df["label"]==label],outlier_columns)
for col in outlier_columns:
    plot_binary_outliers(dataset=dataset, col=col, outlier_col="outlier_lof",reset_index=True)

# --------------------------------------------------------------
# Choose method and deal with outliers
# --------------------------------------------------------------

# Test on single column
col = "gyr_z"
dataset = mark_outliers_chauvenet(df,col=col)
dataset[dataset["gyr_z_outlier"]]
dataset.loc[dataset["gyr_z_outlier"], "gyr_z"] = np.nan

# Create a loop
outliers_removed_df = df.copy()
for col in outlier_columns:
    for label in df["label"].unique():
        dataset = mark_outliers_chauvenet(df[df["label"]==label],col)
        #Replace values marked as outliers with NaN
        dataset.loc[dataset[col + "_outlier"],col] = np.nan
        outliers_removed_df.loc[(outliers_removed_df["label"]==label), col] = dataset[col]
        
        n_outliers = len(dataset) - len(dataset[col].dropna())
        print(f"{label}: {n_outliers} outliers removed from {col}")
# --------------------------------------------------------------
# Export new dataframe
# --------------------------------------------------------------
outliers_removed_df.to_pickle("outliers_removed_chauvenet.pkl")