import pandas as pd
import numpy as np

from data_processing.cgm_data import CGMData
from data_processing.cgm_data_helper import find_standard_meal, get_meal_spike


# Blood sugar after meal should be <180
# https://www.webmd.com/diabetes/how-sugar-affects-diabetes
def max_glucose_dataset(
    cgm_data: CGMData,
    participant_num: int,
    glu_thresh: int = 180,
    hours_after_meal: int = 2,
) -> pd.DataFrame:
    part = cgm_data[participant_num]
    # get the relevant series from other time periods
    grouped_meals = part.food.groupby("time_begin")[
        ["calorie", "total_carb", "dietary_fiber", "sugar", "protein", "total_fat"]
    ].sum(min_count=1)
    # Exclude meals that are close to other meals
    grouped_meals["recent_meals"] = (
        grouped_meals.rolling(
            window=pd.Timedelta(hours=2 * hours_after_meal),
            center=True,
            closed="neither",
        )
        .count()
        .iloc[:, 0]
    )
    grouped_meals = grouped_meals[grouped_meals["recent_meals"] == 1]
    grouped_meals["participant"] = participant_num

    grouped_meals["max_glucose"] = 0
    for start in grouped_meals.index:
        grouped_meals.loc[start, "max_glucose"] = get_meal_spike(
            part.glu, start, hours_after_meal
        ).max()

    grouped_meals["high_glucose"] = grouped_meals["max_glucose"] >= glu_thresh

    grouped_meals.index = range(len(grouped_meals))

    return grouped_meals


# %%
def max_glucose_between_meals_dataset(
    cgm_data: CGMData,
    participant_num: int,
    glu_thresh: int = 180,
    hours_between_meals: int = 2,
) -> pd.DataFrame:
    part = cgm_data[participant_num]
    # get the relevant series from other time periods
    grouped_meals = part.food.groupby("time_begin")[
        ["calorie", "total_carb", "dietary_fiber", "sugar", "protein", "total_fat"]
    ].sum(min_count=1)

    # Exclude meals that are close to other meals
    grouped_meals["recent_meals"] = (
        grouped_meals.rolling(
            window=pd.Timedelta(hours=2 * hours_between_meals),
            center=True,
            closed="neither",
        )
        .count()
        .iloc[:, 0]
    )
    grouped_meals = grouped_meals[grouped_meals["recent_meals"] == 1]
    grouped_meals["participant"] = participant_num

    # Get the maximum glucose reading after a meal before the next meal
    # Idea from: Steven Gubkin
    grouped_meals["max_glu_post_meal"] = 0
    for window in grouped_meals[::-1].rolling(window=2):
        window = window[::-1]
        start = window.index[0]
        if len(window) == 2:
            end = window.index[1]
        else:
            end = None
        grouped_meals.loc[start, "max_glu_post_meal"] = part.glu.loc[start:end].max()[
            "glucose"
        ]

    # Get the glucose reading at the time of the first meal
    grouped_meals["glu_at_first_meal"] = 0
    for window in grouped_meals[::-1].rolling(window=2):
        window = window[::-1]
        start = window.index[0]
        if len(window) == 2:
            end = window.index[1]
        else:
            end = None
        grouped_meals.loc[start, "glu_at_first_meal"] = part.glu.loc[start:end].iloc[0][
            "glucose"
        ]

    # Get the glucose reading at the time of the next meal
    grouped_meals["glu_at_next_meal"] = 0
    for window in grouped_meals[::-1].rolling(window=2):
        window = window[::-1]
        start = window.index[0]
        if len(window) == 2:
            end = window.index[1]
        else:
            end = None
        grouped_meals.loc[start, "glu_at_next_meal"] = part.glu.loc[start:end].iloc[-1][
            "glucose"
        ]

    grouped_meals["high_glucose"] = grouped_meals["max_glu_post_meal"] >= glu_thresh

    grouped_meals.index = range(len(grouped_meals))

    return grouped_meals
