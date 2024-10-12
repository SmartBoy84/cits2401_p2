""" program start point"""

### notes to self

## boolean indexing
# this funky indexing that numpy and pandas allow
# is called: "boolean indexing"
# applying a comparison op yields an array of indices
# passing that array as an index to the original array gets the values
# for which the condition was satisfied (i.e., indices of all True's)
# pandas and numpy have a shared implementations of boolean indexing

## static typing
# I've explicity stated types wherever possible
# this makes the editor be a bit nicer (nice colours!)
# it also makes function nature a bit more obvious

# felt it would be cleaner and more legible separating the different sub-tasks
# into different functions

# furthermore, I feel these function names are the best description
# naming them according to their actual operation is pointless, given their use is
# as the solutions for these parts -> won't be reused elsewhere for their purpose

import pandas as pd

from tasks.task3 import multivariate
from tasks import task1, task2, task3, task4


def main(filename):
    """function core"""

    df = pd.read_csv(filename)

    # task 1
    var, median_result, corr, pca = task1(df)
    print(var, median_result, corr, pca.shape, pca[:5])

    # task 2
    df = df.apply(pd.to_numeric, errors="coerce")  # replace non-ints with NaNs
    task2i, task2ii = task2(df)
    print(task2i[:4], task2ii)

    # task 3
    (task3i,) = task3(df)  # (task3,) needed to exhaust returned generator
    # print(task3i[0][0], task3i[1][0])
    # print(multivariate())

    # task 4
    # task4(df, pca)
    multivariate()


main("data/weather_data.csv")
