""" setup tasks module """

import warnings

from .task1 import Task1
from .task2 import Task2
from .task3 import Task3
from .task4 import Task4


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
    """task 4"""
    # can pass True to graph plots on separate figures (windows)
    return Task4(df, pca, True).get_output()
