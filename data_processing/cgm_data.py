from typing import Dict
import numpy as np
import pandas as pd

from data_processing.constants import ACC_G_MODE, SMALL_G
from data_processing.cgm_data_helper import acc_high_pass

DATA_PATH = "data"


class Patient:

    metric_map = {
        "acc": "ACC",
        "bvp": "BVP",
        "glu": "Dexcom",
        "eda": "EDA",
        "food": "Food_Log",
        "hr": "HR",
        "ibi": "IBI",
        "temp": "TEMP",
    }

    def __init__(self, patient_id, gender, hba1c) -> None:
        self.patient_id: int = patient_id
        self.gender: str = gender
        self.hba1c: float = hba1c

        self._acc: pd.DataFrame = None
        self._acc_tot: pd.DataFrame = None
        self._bvp: pd.DataFrame = None
        self._glu: pd.DataFrame = None
        self._eda: pd.DataFrame = None
        self._food: pd.DataFrame = None
        self._hr: pd.DataFrame = None
        self._ibi: pd.DataFrame = None
        self._temp: pd.DataFrame = None

    def get_file_path(self, metric):
        metric_name = Patient.metric_map[metric]
        return f"./{DATA_PATH}/{self.patient_id:03d}/{metric_name}_{self.patient_id:03d}.csv"

    def get_gi_path(self):
        return f"./data_processing/Glycemic_Index/GI_{self.patient_id:d}.csv"

    @property
    def acc(self):
        if self._acc is None:
            self._acc = pd.read_csv(
                self.get_file_path("acc"),
                index_col=["datetime"],
                parse_dates=["datetime"],
                engine="pyarrow",
            )
            self._acc.columns = [c.strip() for c in self._acc.columns]
            # Convert all columns to m/s^2 from int8 in 2G mode
            # TODO It is possible that values <0 should be divided by 128 and > 0 by 127
            self._acc["acc_x"] = (self._acc["acc_x"] * ACC_G_MODE * SMALL_G) / 127
            self._acc["acc_y"] = (self._acc["acc_y"] * ACC_G_MODE * SMALL_G) / 127
            self._acc["acc_z"] = (self._acc["acc_z"] * ACC_G_MODE * SMALL_G) / 127
        return self._acc

    @property
    def acc_tot(self):
        if self._acc_tot is None:
            acc_filt = acc_high_pass(self.acc, 0.5, None, None)
            self._acc_tot = pd.DataFrame(
                {
                    "acc": np.sqrt(
                        acc_filt["acc_x"] ** 2
                        + acc_filt["acc_y"] ** 2
                        + acc_filt["acc_z"] ** 2
                    )
                }
            )
        return self._acc_tot

    @property
    def bvp(self):
        if self._bvp is None:
            self._bvp = pd.read_csv(
                self.get_file_path("bvp"),
                index_col=["datetime"],
                parse_dates=["datetime"],
                engine="pyarrow",
            )
            self._bvp.columns = [c.strip() for c in self._bvp.columns]
        return self._bvp

    @property
    def glu(self):
        if self._glu is None:
            self._glu = pd.read_csv(
                self.get_file_path("glu"),
                header=0,
                skiprows=range(1, 13),
                index_col=["Timestamp (YYYY-MM-DDThh:mm:ss)"],
                parse_dates=["Timestamp (YYYY-MM-DDThh:mm:ss)"],
            )
            self._glu = self._glu[self._glu["Event Type"] == "EGV"]
            self._glu = self._glu.rename(columns={"Glucose Value (mg/dL)": "glucose"})
            self._glu = self._glu.rename_axis("datetime")
            self._glu = self._glu[["glucose"]]
        return self._glu

    @property
    def eda(self):
        if self._eda is None:
            self._eda = pd.read_csv(
                self.get_file_path("eda"),
                index_col=["datetime"],
                parse_dates=["datetime"],
                engine="pyarrow",
            )
            self._eda.columns = [c.strip() for c in self._eda.columns]
        return self._eda

    def _load_food(self):
        food_path = self.get_file_path("food")
        gi_path = self.get_gi_path()

        gi_df = pd.read_csv(
            gi_path, index_col=["time_begin"], parse_dates=["time_begin"]
        )

        if self.patient_id == 3:
            # Patient 3 has no header
            self._food = pd.read_csv(food_path, skipinitialspace=True)
            self._food.columns = [
                "date",
                "time",
                "time_begin",
                "logged_food",
                "amount",
                "unit",
                "searched_food",
                "calorie",
                "total_carb",
                "sugar",
                "protein",
            ]
            self._food["time_begin"] = pd.to_datetime(self._food["time_begin"])
            self._food["time_end"] = None
            self._food = self._food.set_index("time_begin")
            self._food["total_fat"] = np.nan
            self._food["dietary_fiber"] = np.nan
        else:
            self._food = pd.read_csv(
                food_path,
                index_col=["time_begin"],
                parse_dates=["time_begin"],
                skipinitialspace=True,
            )

        self._food["time_end"] = pd.to_datetime(
            self._food["date"] + " " + self._food["time_end"]
        )

        end_times = self._food.groupby("time_begin")["time_end"].min()
        self._food = self._food.merge(
            end_times,
            how="left",
            on="time_begin",
            suffixes=("", "_x"),
            validate="many_to_one",
        )
        self._food["time_end"] = self._food["time_end"].fillna(self._food["time_end_x"])
        self._food = self._food.drop("time_end_x", axis=1)

        # Fill in NaN in searched food with empty string
        self._food["searched_food"] = self._food["searched_food"].fillna("")

        # Some food logs have `time_of_day` instead of `time`
        if self.patient_id in [7, 13, 15, 16]:
            self._food = self._food.drop(["date", "time_of_day"], axis=1)
        else:
            self._food = self._food.drop(["date", "time"], axis=1)

        self._food["gi"] = gi_df["GI"]
        self._food["gl"] = self._food["total_carb"] * self._food["gi"] * 0.01

    @property
    def food(self):
        if self._food is None:
            self._load_food()
        return self._food

    @property
    def hr(self):
        if self._hr is None:
            # Patient 1 has no seconds recorded
            if self.patient_id == 1:
                self._hr = pd.read_csv(
                    self.get_file_path("hr"),
                    parse_dates=["datetime"],
                    date_format="%m/%d/%y %H:%M",
                    engine="pyarrow",
                )
                self._hr.loc[
                    self._hr.groupby("datetime").cumcount() + 1 != 1, "datetime"
                ] = pd.NaT
                self._hr = self._hr.interpolate(method="linear")
                self._hr = self._hr.set_index("datetime")
            else:
                self._hr = pd.read_csv(
                    self.get_file_path("hr"),
                    index_col=["datetime"],
                    parse_dates=["datetime"],
                    engine="pyarrow",
                )
            self._hr.columns = [c.strip() for c in self._hr.columns]

        return self._hr

    @property
    def ibi(self):
        if self._ibi is None:
            self._ibi = pd.read_csv(
                self.get_file_path("ibi"),
                index_col=["datetime"],
                parse_dates=["datetime"],
                engine="pyarrow",
            )
            self._ibi.columns = [c.strip() for c in self._ibi.columns]
        return self._ibi

    @property
    def temp(self):
        if self._temp is None:
            self._temp = pd.read_csv(
                self.get_file_path("temp"),
                index_col=["datetime"],
                parse_dates=["datetime"],
                engine="pyarrow",
            )
            self._temp.columns = [c.strip() for c in self._temp.columns]
        return self._temp


class CGMData:
    patients: Dict[int, Patient] = {}
    demographics: pd.DataFrame = None
    _instance = None
    _loaded = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not self._loaded:
            self.demographics = pd.read_csv(
                f"./{DATA_PATH}/Demographics.csv", index_col="ID"
            )
            self.__class__._loaded = True

    def __getitem__(self, key: int):
        if key < 1 or key > 16:
            raise IndexError(f"Patient ID {key} out of range")
        if key not in self.patients:
            patient_demo = self.demographics.loc[key]
            self.patients[key] = Patient(
                key, patient_demo["Gender"], patient_demo["HbA1c"]
            )
        return self.patients[key]
