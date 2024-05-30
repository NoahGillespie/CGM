import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from data_processing.constants import ACC_HZ

## Data Manipulation


# Get the curve immediately after food consumption
def get_meal_spike(glu: pd.DataFrame, time, duration: int):
    # get a slice of glu for duration hours after time
    glu_slice = glu.loc[time : time + pd.Timedelta(hours=duration), "glucose"]
    glu_slice.index = glu_slice.index - glu_slice.index[0]
    # sometimes the measurements are off by a second
    glu_slice.index = glu_slice.index.round(freq="5min")
    return glu_slice


def find_standard_meal(df: pd.DataFrame):
    bfast_filter = df["searched_food"].str.contains("Partly Skimmed Milk") | df[
        "searched_food"
    ].str.contains("Frosted Flakes")
    return df[bfast_filter]


def acc_high_pass(acc, cutoff_freq, time_start, time_end):
    acc_slice = acc.loc[time_start:time_end]
    res = acc_slice.copy()
    b, a = butter(4, 2 * cutoff_freq / ACC_HZ, btype="high")
    for col in res:
        res[col] = filtfilt(b, a, acc_slice[col])
    return res


## Plotting


# Plot the series with food consumption marked
def plot_series_with_food(series, food, time_start, time_end, glu_col_name="glucose"):
    series_slice = series.loc[time_start:time_end, glu_col_name]
    series_slice_range = series_slice.max() - series_slice.min()
    series_slice_mid = (series_slice.max() + series_slice.min()) / 2
    food_slice = food[
        (time_start <= food.index)
        & (food.index <= pd.Timestamp(time_end) + pd.Timedelta(days=1))
    ]

    ymin = series_slice_mid - series_slice_range * 0.6
    ymax = series_slice_mid + series_slice_range * 0.6

    plt.figure(figsize=(20, 4))
    plt.plot(series_slice)
    plt.vlines(food_slice.index, ymin=ymin, ymax=ymax, colors="red")


# Plot the series with food consumption marked
def plot_series_with_diff(series, food, time_start, time_end, glu_col_name="glucose"):
    series_slice = series.loc[time_start:time_end, glu_col_name].diff()
    series_slice_range = series_slice.max() - series_slice.min()
    series_slice_mid = (series_slice.max() + series_slice.min()) / 2
    food_slice = food[
        (time_start <= food.index)
        & (food.index <= pd.Timestamp(time_end) + pd.Timedelta(days=1))
    ]

    ymin = series_slice_mid - series_slice_range * 0.6
    ymax = series_slice_mid + series_slice_range * 0.6

    plt.figure(figsize=(20, 4))
    plt.plot(series_slice)
    plt.vlines(food_slice.index, ymin=ymin, ymax=ymax, colors="red")
