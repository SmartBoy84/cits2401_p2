""" solutions for task 2"""

from pandas import DataFrame

from .task import Task
from .task1 import Task1 


class Task2(Task):
    """task 2"""

    def __init__(self, df: DataFrame) -> None:
        super().__init__(df, (self.part_i, self.part_ii))

    def part_i(self):
        """solution for part 2.i"""
        mean_apparent_temp = self.df["Apparent Temperature (C)"].mean()

        # filter entires for which rain is yes and temp is less than mean apparent temp
        filtered_df = self.df[
            (self.df["Rain"] == Task1.RAIN_MAP["Yes"])  # from part 1.iv ...
            & (self.df["Temperature (C)"] < mean_apparent_temp)
        ]["Temperature (C)"]

        # round temperatures to 2 decimal places
        return [round(temp, 2) for temp in filtered_df]

    def part_ii(self):
        """solution for part 2.ii"""
        if self.df.isna().values.any():
            self.df.fillna(0, inplace=True)
            return True

        return False
