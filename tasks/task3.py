""" solutions for task 1"""

from pandas import DataFrame
import sympy as sp

from .task import Task


def multivariate():
    pass


class Task3(Task):
    """task 3"""

    # make it a constant class variable
    EXPR = sp.diff(sp.parse_expr("2.5*x**3 + 3*x**2 + 3.5*x + 5"))

    def __init__(self, df: DataFrame):
        super().__init__(df, (self.part_i,))

    def part_i(self) -> list[float]:
        """part 3.i"""

        # note, experimented with applying vectorised numpy functions
        # was disappointed - were much slower as numpy.array needed to be allocated (I assume...)

        # pandas automatically determines the set of values under rain and groups according to them
        # i.e., puts all rows with that value under same dataframe!
        yes_no_records = self.df.groupby("Rain")["Temperature (C)"]

        # apply() applies a function to each entry (at the first level)
        # since yes_no_records df will have "two" values (since Rain=1/0)
        # it will run the function twice
        # the function is a list comprehension which substitutes
        # temps into the derivative expression and rounds to 2 decimals places
        temp_derivatives = yes_no_records.apply(
            lambda group: [round(self.EXPR.subs("x", temp), 2) for temp in group]
        )

        # .tolist() needed as expected return type is list (refer to fn signature)
        return temp_derivatives.tolist()
