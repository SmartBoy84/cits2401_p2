"""
Author: Hamdan Mahmood
Student Number: 24275627
Libraries used: SymPy, NumPy, Pandas, MatplotLib

This project performs data analysis on a set of weather records
to derive potential correlations to build reccomendations off

For clarity, each task's components have been grouped under the same class
Each Task[n] class inherits from the parent Task class
This parent class provides convenient methods to run the functions under each class
to create output in the format requested by the assignment sheet

In addition, Task4 provides various plotting functions for the weather dataset
The user is able to select whether to plot them all on one screen or multiple
This behaviour can be changed by toggling to boolean under the task4 function

NOTE: the df passed into task4 must be unaltered 
(i.e., str entries must NOT have been mapped)
"""

import pandas as pd
import numpy as np
import sympy as sp

from matplotlib import pyplot as plt, colors as mcolors
from pandas import DataFrame


# ---------------- Parent task class ----------------
class Task:
    """
    primary task class
    this allowed me to experiment with general task functions
    that aided in development and debugging
    """

    def __init__(self, df: DataFrame, tasks=()):
        # do all assignments here
        # purpose of this is a sort of abstraction for the tasks

        self.df = df
        self.tasks = tasks

    def get_output(self):
        """generate output for task"""

        # I did this classes stuff as practise
        # feel like it turned out quite well!

        # tuple([]) to force interpreter to run task,
        # even if it's somehow found out that it returns None
        return tuple([task() for task in self.tasks])


# ------------------------ Task 1 ---------------------------
class Task1(Task):
    """Task 1"""

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


# ------------------------ Task 2 ---------------------------
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


# ------------------------ Task 3 ---------------------------
def multivariate():
    """solve sle x = y^0.25 + z^0.34"""
    # Define Y and Z as symbols
    y, z = sp.symbols("Y Z")

    # create expression
    x = y**0.25 + z**0.34

    # differentiate w.r.t x and y
    dx_dy = sp.diff(x, y)
    dx_dz = sp.diff(x, z)

    # try to solve sle dx/dy = 0 and dx/dz = 0
    # the fact that we had to do this was clarified by andrei ristea
    solutions = sp.solve([dx_dy, dx_dz], (y, z))

    print(solutions)


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


# ------------------------ Task 4 ---------------------------
class Task4(Task):
    """task 4"""

    ### Settings
    SINGLE_SCREEN_SETTINGS = {
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.labelweight": "bold",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 10,
        "lines.markersize": 1.1,
    }

    BOX_PLOT_FONT_OVERRIDE = 8

    BOX_PLOT_MAX_CHARS = 10
    # max chars before new line for box plot
    # I do this to avoid having to rotate=90 or set font size to like 5
    # to fit labels

    # everything needs to be hardcoded due to unorthodox plotting
    PIE_RADIUS = 1.1
    PIE_ANCHOR = (0.5, 0.8)  # achor the pie plot withing subplot
    PIE_LEGEND_ANCHOR = (0.5, -0.22)  # where to position legend (relative to loc)
    PIE_TITLE_PAD = 32  # hardcoded to make it in line with other titles

    ## Global configuration
    GENERAL_COLOR = (1, 179 / 255, 64 / 255)
    # https://matplotlib.org/stable/users/explain/colors/colors.html#colors-def

    # viridis is a color map - specifically designed to improve readability
    CMAP = plt.get_cmap("viridis")

    SINGLE_SCREEN_SIZE = (14, 6.8)  # figsize for when single figure plot (plot_all)

    GRAPH_SIZE = (9, 5)  # figsize for graphs (part 1,2,4,5)
    PIECHART_SIZE = (6, 6)  # figsize for piechart

    def __init__(self, df, pca: DataFrame, one_screen_display=True):
        plt.rcParams.update(self.SINGLE_SCREEN_SETTINGS)  # apply settings
        self.one_screen_display = one_screen_display

        if one_screen_display:
            # woah! can use the same master task!
            super().__init__(df, (self.plot_all,))
        else:
            # default axes for plot functions is plt -
            # if run individually will create separate screens
            super().__init__(
                df,
                (
                    self.part_i,
                    self.part_ii,
                    self.part_iii,
                    self.part_iv,
                    self.part_v,
                    plt.show,
                    # last one needed to prevent blocking
                    # (blocking occurs if plt.show run in each function)
                ),
            )

        self.pca = pca

    def plot_all(self):
        """display all plots on one screen"""
        # _, axs = plt.subplots(2, 3, figsize=self.SINGLE_SCREEN_SIZE)  # 2x3 (w x l)
        fig = plt.figure(figsize=self.SINGLE_SCREEN_SIZE)
        gs = fig.add_gridspec(2, 3)
        # create a grid
        # this allows a bit more freedom with spacing than .subplots
        # since I want the pieplot to take up two subplots

        # graphs
        self.part_i(fig.add_subplot(gs[0, 0]))  # top-left
        self.part_iii(fig.add_subplot(gs[1, 0]))  # bottom-left
        self.part_iv(fig.add_subplot(gs[0, 1]))  # top-right
        self.part_v(fig.add_subplot(gs[1, 1]))  # bottom-right

        # merge rightmost cells
        ax_pie = fig.add_subplot(gs[:, 2])  # Right side (merged area for pie chart)

        # pie chart - will span two spaces to accomadate legend (more to make use of the free space)
        self.part_ii(ax_pie)  # Spanning pie chart

        plt.tight_layout()  # prevent overlap
        plt.show()  # display figure

    def part_i(self, ax=None):
        """part i"""

        filtered_records = self.df[
            (self.df["Wind Speed (km/h)"] < 10) & (self.df["Visibility (km)"] > 9)
        ]

        if ax is None:
            # took way too long to figure this out...
            # plt.figure call BEFORE plt.gca else same figure used for two plots
            plt.figure(figsize=self.GRAPH_SIZE)
            ax = plt.gca()

        ax.hist(
            filtered_records["Temperature (C)"],
            bins=20,
            color=self.GENERAL_COLOR,
            edgecolor="black",
        )

        # ugh, quite cumbersome but pyplot doesn't have defaults
        ax.set_title("Temperature Distribution [Wind Speed < 10 & Visibility > 9]")
        ax.set_xlabel("Temperature (C)")
        ax.set_ylabel("Frequency")

    def part_ii(self, ax=None):
        """part ii"""

        no_rain_entries = self.df[self.df["Rain"] == "No"]
        summary_groups = no_rain_entries.groupby("Summary")

        temp_sums = summary_groups["Temperature (C)"].sum()

        labels = temp_sums.index

        if ax is None:
            plt.figure(figsize=self.PIECHART_SIZE)
            ax = plt.gca()

        if self.one_screen_display:
            free_spaces, _, autopct = ax.pie(
                temp_sums,
                autopct="%.1f%%",  # display wedge size
                explode=[0.04] * len(temp_sums),
                radius=self.PIE_RADIUS,
                # pyplot has weird spacing calculation formule, easiest to hardcode
            )
        else:
            free_spaces, _, autopct = ax.pie(
                temp_sums,
                labels=labels,
                autopct="%.1f%%",  # display wedge size
                explode=[0.03] * len(temp_sums),
                radius=0.8,
                # pyplot has weird spacing calculation formule, easiest to hardcode
            )

        ax.set_anchor(self.PIE_ANCHOR)

        plt.setp(autopct, fontweight="bold", fontsize=9)
        # percentage font size and weight

        # create legend and title
        title = "Temperature Distribution by Summary [Rain = No]"

        if self.one_screen_display:
            ax.legend(
                free_spaces, labels, loc="center", bbox_to_anchor=self.PIE_LEGEND_ANCHOR
            )
            ax.set_title(title, pad=self.PIE_TITLE_PAD)
        else:
            ax.set_title(title)

        # ax.set_position((0.42, 0.25))  # Adjust to center it

    def part_iii(self, ax=None):
        """part iii"""

        summary_groups = self.df.groupby("Summary")
        humidities = summary_groups["Humidity"].apply(list)
        # long adieu later... turns out groupby() returns some intermediate structure
        # to which an aggregator function has to be applied

        labels = humidities.index

        if ax is None:
            plt.figure(figsize=self.GRAPH_SIZE)
            ax = plt.gca()

        # I really don't want to rotate 90 degrees
        # so I'm going to add new line to fit labels
        multiline_labels = []
        for label in labels:
            final_word = ""
            count = 0
            for word in label.split(" "):
                count += len(word)
                if count > self.BOX_PLOT_MAX_CHARS:
                    final_word += (
                        "\n"  # next word would exceed max char set so add new line
                    )
                    count = 0
                final_word += f" {word}"
            multiline_labels.append(final_word[1:])  # [1:] to exclude first space

        # create graph
        patches = ax.boxplot(
            humidities,
            tick_labels=multiline_labels,
            boxprops={"color": "black", "linewidth": 1.5},
            medianprops={"color": "black"},
            patch_artist=True,
        )

        # https://matplotlib.org/stable/gallery/statistics/boxplot_color.html
        for patch in patches["boxes"]:
            patch.set_facecolor(self.GENERAL_COLOR)

        ax.set_title("Humidity Spread for Different Summaries")

        ax.set_xlabel("Summary", wrap=True)
        ax.set_ylabel("Humidity")

        plt.setp(ax.get_xticklabels(), fontsize=self.BOX_PLOT_FONT_OVERRIDE)
        # ax.suptitle("")

    def part_iv(self, ax=None):
        """part iv"""

        temp_matrix = self.df[
            ["Temperature (C)", "Apparent Temperature (C)"]
        ].to_numpy()

        if ax is None:
            plt.figure(figsize=self.GRAPH_SIZE)
            ax = plt.gca()

        ax.plot(
            temp_matrix[:, 0],
            temp_matrix[:, 1],
            color=self.GENERAL_COLOR,
            marker="o",
            linestyle="none",
            # I don't want to do scatter as that doesn't have
            # convenience methods to set colour of all points
        )

        ax.set_title("Plot of Temperature against Apparent Temperature")

        ax.set_xlabel("Temperature (C)")
        ax.set_ylabel("Apparent Temperature (C)")

    def part_v(self, ax=None):
        """part v"""
        # note; I did histogram instead of bar plot as bar plot conveys no information
        # for continuous scales (CONFIRM WITH LECTURER!)

        ## my personal colour code algorithm:
        # sum the pca values for all the wind speeds included under a given histogram bar
        # normal the pca value array and map it to the chosen (viridis) color map

        filtered_indices = self.df["Wind Speed (km/h)"] > 8
        wind_speeds = self.df[filtered_indices]["Wind Speed (km/h)"]

        bin_n = 20
        bin_edges = np.linspace(min(wind_speeds), max(wind_speeds), bin_n)

        if ax is None:
            plt.figure(figsize=self.GRAPH_SIZE)
            ax = plt.gca()

        _, _, patches = ax.hist(
            wind_speeds,
            bins=bin_edges,
            color=self.GENERAL_COLOR,
            edgecolor="black",
        )

        # set histogram patch colors
        pca_vals = self.pca[filtered_indices]

        # get pca sums for each bar
        patch_pca = [
            pca_vals[wind_speeds.between(bin_edges[i], bin_edges[i + 1])].sum()
            for i in range(bin_n - 1)
        ]

        # normalising it increases coherence of output colors - makes sure they are related in some way
        norm = plt.Normalize(min(patch_pca), max(patch_pca))
        colors = [mcolors.to_hex(self.CMAP(norm(value))) for value in patch_pca]

        for patch, color in zip(patches, colors):
            patch.set_facecolor(color)

        # marker: here is my barplot implementation - to me, it seems to convey no info so I've foregone it
        # norm = plt.Normalize(min(self.pca), max(self.pca))
        # colors = [mcolors.to_hex(self.CMAP(norm(value))) for value in self.pca]

        # ax.bar(wind_speeds.index, wind_speeds, color = colors)

        ax.set_title("Histogram of Wind Speeds (km/h) > 8")

        ax.set_xlabel("Wind Speed (km/h)")
        ax.set_ylabel("Frequency")


# ------------------------ Output functions ---------------------------
def task1(df):
    """task 1"""
    return Task1(df).get_output()


def task2(df):
    """task 2"""
    return Task2(df).get_output()


def task3(df):
    """task 3"""
    return Task3(df).get_output()


def task4(df, pca):
    """
    task 4
    note: the df passed into this task cannot have strings mapped to integers!
    i.e., the df should not have passed through the earlier tasks
    """
    # can pass True to graph plots on separate figures (windows)
    return Task4(df, pca, True).get_output()


# ------------------------ start point---------------------------
def main(filename):
    """function core"""

    df = pd.read_csv(filename)

    # task 1
    var, median_result, corr, pca = task1(df)

    # task 2
    df = df.apply(pd.to_numeric, errors="coerce")  # replace non-ints with NaNs
    task2i, task2ii = task2(df)

    # task 3
    (task3i,) = task3(df)  # (task3,) needed to exhaust returned generator

    return (var, median_result, corr, pca, task2i, task2ii, task3i)


fileName = "data/weather_data.csv"

var, medianResult, corr, pca, task2i, task2ii, task3i = main(fileName)

task4(pd.read_csv(fileName), pca)

# multivariate()

# print(var, medianResult, corr, pca.shape, pca[:5])
# print(task2i[:4], task2ii)
# print(task3i[0][0], task3i[1][0])
