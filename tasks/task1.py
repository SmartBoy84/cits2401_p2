""" solutions for task 1"""

import numpy as np
from pandas import DataFrame

from .task import Task

class Task1(Task):
    """task 1"""

    SUMMARY_MAP = {
        "Partly Cloudy": 1,
        "Mostly Cloudy": 2,
        "Overcast": 3,
        "Foggy": 4,
        "Clear": 5,
        "Breezy and Mostly Cloudy": 6,
    }

    RAIN_MAP = {"Yes": 1, "No": 2}

    def __init__(self, df: DataFrame):
        super().__init__(df, (self.part_i, self.part_ii, self.part_iii, self.part_iv))

    def part_i(self) -> float:
        """solution to 1.i"""

        mostly_cloudy_entries = self.df[self.df["Summary"] == "Mostly Cloudy"]
        humidities = np.array(mostly_cloudy_entries["Humidity"])

        mean_humidity = np.sum(humidities) / len(humidities)

        # note: am aware of the existence of .var() but felt manual implementation would be better
        variance = np.sum(((humidities - mean_humidity) ** 2)) / len(humidities)

        return float(round(variance, 2))

    def part_ii(self) -> list:
        """solution to 1.ii"""

        will_rain_entries = self.df[
            self.df["Rain"] == "Yes"
        ]  # runs before Rain entries mapped to int...
        will_rain_temps = np.array(will_rain_entries["Temperature (C)"])

        ## old implementation
        # sorted_will_rain_temps = np.sort(np.array(will_rain_entries["Temperature (C)"]))

        # lower_percentile_temps = sorted_will_rain_temps[
        #     : (
        #         math.ceil(0.25 * len(sorted_will_rain_temps))
        #     )
        # ]
        # upper_percentile_temps = sorted_will_rain_temps[
        #     math.ceil(round(0.75 * len(sorted_will_rain_temps)))
        #     - 1 :
        # ]

        ## new implementation using numpy methods
        # lower percentile median
        percentile_lower = np.percentile(will_rain_temps, 25)
        lower_percentile_temps = will_rain_temps[will_rain_temps <= percentile_lower]
        lower_percentile_median = np.median(lower_percentile_temps)

        # upper percentile median
        percentile_upper = np.percentile(will_rain_temps, 75)
        upper_percentile_temps = will_rain_temps[will_rain_temps >= percentile_upper]
        upper_percentile_median = np.median(upper_percentile_temps)

        # note to marker: a bit of doc searching led me to percentile() method
        # I felt this was clearer than my "manual" method to find percentiles so I used this instead
        # old implementation is still included to show conceptual understanding

        # convert from np.f64 -> float and round to 2 decimal places
        return list(
            float(round(temp, 2))
            for temp in [lower_percentile_median, upper_percentile_median]
        )

    def part_iii(self) -> float:
        """solution to 1.iii"""

        # extract filtered df
        filtered_df = self.df[
            (self.df["Summary"] == "Mostly Cloudy")
            & (self.df["Temperature (C)"] > self.df["Apparent Temperature (C)"])
        ]

        dot_product_vectors = (
            filtered_df["Wind Speed (km/h)"] * filtered_df["Wind Bearing (degrees)"]
        )
        correlation = dot_product_vectors.corr(filtered_df["Visibility (km)"])

        return float(round(correlation, 2))

    def part_iv(self):
        """part iv solution"""

        ## convert string -> ints
        # note, origiinal df needs to be altere here as example output for task2ii is false

        # summary mapping - done after matrix creation to preserve original DataFrame
        self.df["Summary"] = self.df["Summary"].map(
            self.SUMMARY_MAP
        )  # map can also take a function - this is just an explicit mapping

        # rain status mapping
        self.df["Rain"] = self.df["Rain"].map(self.RAIN_MAP)

        ## 1- extract the relevant columns, in the relevant order
        extracted_df = self.df[
            [
                "Summary",
                "Rain",
                "Temperature (C)",
                "Apparent Temperature (C)",
                "Humidity",
            ]
        ]

        ## a) standardize df
        standardised_df = (extracted_df - extracted_df.mean()) / extracted_df.std()
        # so cool! what's happening here is similar to boolean indexing
        # also note that pandas automatically casts string -> int (if possible)

        ## b) determine covariance matrix
        covariance_matrix = standardised_df.cov()

        ## c) get eigenvectors of covariance matrix
        (eigen_values, eigen_vectors) = np.linalg.eig(covariance_matrix)
        # perform eigen decomposition

        ## d) sort eigenvectors from highest -> lowest
        eigen_vectors = eigen_vectors[np.argsort(-eigen_values)]
        # argsort() returns indices of sorted entries of eigen_values
        # => used to index eigen_vectors

        ## e) select top k (1) eigenvectors
        top_k_eigen_vectors = eigen_vectors[:, 0]

        ## f) transform df using selected eigenvectors
        transformed_df = standardised_df.dot(top_k_eigen_vectors)  # compute dot product

        # I emailed Andrei Ristea, their first value was 1.3 - I get 1.29
        # I could implement a hacky solution for forced ceiling when at 0.5
        # to get my output to match theirs but I'm going to leave it as is
        # since this is the default python behaviour (Bankers rounding)
        return transformed_df.round(2).to_numpy()
