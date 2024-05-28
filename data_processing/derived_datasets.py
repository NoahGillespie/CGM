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