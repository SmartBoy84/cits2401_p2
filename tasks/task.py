"""General task layout"""

from pandas import DataFrame


class Task:
    """task"""

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
