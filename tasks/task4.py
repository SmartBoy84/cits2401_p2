""" solutions for task 4"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from pandas import DataFrame

from .task import Task


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

        no_rain_entries = self.df[self.df["Rain"] == "Yes"]
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
