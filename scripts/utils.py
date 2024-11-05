import os
import math
import statistics
from pathlib import Path
import pandas as pd


def load_csv_dataset(csv_path: os.PathLike, drop_nan: bool = True) -> pd.DataFrame:
    """
    Load a dataset from a csv file and return a dataframe
    Args:
        csv_path: path to the csv file
        drop_nan: True to remove all rows containing Nan values from the dataframe

    Returns: dataframe
    """
    file_path = Path(csv_path)
    try:
        data = pd.DataFrame(pd.read_csv(file_path)).dropna()
        return data.dropna() if drop_nan else data
    except FileNotFoundError as e:
        print(f"Could not find data in specified path: {file_path}.")
        raise e


def load_bins_from_file(filepath: os.PathLike) -> list:
    """
    Read a text file containing comma separated bin cut-offs in a form that looks something like:

    1, 3, 5, 7, 9, 11...

    Args:
        filepath: path to the text file containing bin cut-offs

    Returns: list of integers of bin cut-offs
    """
    with open(filepath, "r") as f:
        data = f.read()
        f.close()
    # it would be nice to do some checking here to make sure that the bin cutoffs make sense... maybe someday
    return [int(i.strip()) for i in data.split(",")]


def semivariance_pair(z_near: float, z_far: float) -> float:
    """
    Calculate semi variance between 2 points
    Args:
        z_near: Value at the near point
        z_far: Value at the far point

    Returns: Semi-variance between the points
    """
    return (z_near - z_far)**2 / 2


def calculate_range(x1: float, x2: float, y1: float, y2: float, z1: float = 0.0, z2: float = 0.0) -> float:
    """
    Calculate the euclidean distance between two points
    Args:
        x1: x location of first point
        x2: x location of second point
        y1: y location of first point
        y2: y location of second point
        z1: z location of first point
        z2: z location of second point

    Returns:
    """
    # note that if we want the range in 2 dimensions, just use z1, z2 = 0 (which is the default).
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)


def create_mean_bins(
        sorted_ranges: list[float],
        sorted_values: list[float],
        bin_cutoffs: list) -> tuple[list, list, list, list]:
    """
    Create a list of mean range values and a list of mean semivariance values according to a list of bins cut-offs.
    Args:
        sorted_ranges: list of ranges, sorted in ascending order
        sorted_values: corresponding list of values, sorted in ascending order
        bin_cutoffs: list of cut off ranges

    Returns: a list of mean ranges and a list of mean values.
    """
    range_means = []
    gamma_means = []
    gamma_stdevs = []
    bin_counts = []
    last_bin_index = 0
    current_range_index = 0
    # let's use sliding window logic to create the bins
    for i, (r, v) in enumerate(zip(sorted_ranges, sorted_values)):

        # two scenarios captured in first statement here:
        # 1. normal scenario: we are at a range less than the threshold, so continue
        # 2. we have reached the end of the ranges. Stop here ( this is the len - 1 logic) and slice
        if r < bin_cutoffs[current_range_index] and i != len(sorted_ranges) - 1:
            continue
        else:
            # we have reached the limit of this range. slice, then compute the means of the slice and append
            rs = sorted_ranges[last_bin_index:i]  # end index is exclusive, so we use i here
            vs = sorted_values[last_bin_index:i]
            # it's always possible that a bin is empty:
            if bool(rs) and bool(vs):
                m_rs = statistics.mean(rs)
                m_vs = statistics.mean(vs)
                std_vs = statistics.stdev(vs)
                bin_count = i - last_bin_index
                range_means.append(m_rs)
                gamma_means.append(m_vs)
                gamma_stdevs.append(std_vs)
                bin_counts.append(bin_count)

            # increment counters for next iteration
            last_bin_index = i
            current_range_index += 1

    # in ready to plot form
    print(f"Num bins: {len(range_means)}")
    return range_means, gamma_means, gamma_stdevs, bin_counts


def exponential(max_range: int, w: float, a: float) -> tuple[list, list]:
    """
    Fit an exponential model.
    Args:
        max_range: The maximum range to fit the model for
        w: the sill
        a: the range parameter

    Returns: list of x values and corresponding y values for the fit
    """
    # here we mean y in terms of a plot, not the dimension y in x,y,z space
    y_fit = []
    xs = []

    # this will create a fit at integer range intervals up until the x_range
    for x in range(max_range):
        f = w * (1 - math.e ** -( x / a ))
        xs.append(x)
        y_fit.append(f)

    return xs, y_fit


def spherical(max_range: int, w: float, a: float) -> tuple[list, list]:
    """
    Fit a spherical model.
    Args:
        max_range: The maximum range to fit the model for
        w: the sill
        a: the range parameter

    Returns: list of x values and corresponding y values for the fit
    """
    # here we mean y in terms of a plot, not the dimension y in x,y,z space
    y_fit = []
    xs = []

    # this will create a fit at integer range intervals up until the x_range
    for x in range(max_range):
        if x > a:
            # fit is the sill when out side of the range
            f = w
        else:
            f = w * ((3 / 2) * (x / a) - (1 / 2) * ((x / a) ** 3))
        xs.append(x)
        y_fit.append(f)

    return xs, y_fit


def gaussian(max_range: int, w: float, a: float) -> tuple[list, list]:
    """
    Fit a gaussian model.
    Args:
        max_range: The maximum range to fit the model for
        w: the sill
        a: the range parameter

    Returns: list of x values and corresponding y values for the fit
    """
    # here we mean y in terms of a plot, not the dimension y in x,y,z space
    y_fit = []
    xs = []

    # this will create a fit at integer range intervals up until the x_range
    for x in range(max_range):
        f = w * (1 - math.e ** - ( x / a ) ** 2)
        xs.append(x)
        y_fit.append(f)

    return xs, y_fit


def confidence_interval_model(mean_gammas: list, stdev_gammas: list, num_pairs: list) -> tuple[list, list]:
    """
    Model an upper and lower 95% confidence interval.
    Args:
        mean_gammas: list of the mean of the semivariance for bins
        stdev_gammas: list of standard deviations of the semivariance for bins
        num_pairs: list of number of pairs in each bin

    Returns:

    """
    lower_values = []
    upper_values = []
    for mg, stdev_g, n in zip(mean_gammas, stdev_gammas, num_pairs):
        magnitude = confidence_interval(stdev_g, n)
        lower_values.append(mg - magnitude)
        upper_values.append(mg + magnitude)
    return lower_values, upper_values


def confidence_interval(stdev_gamma: float, num_pairs: int) -> float:
    """
    Calculate a 95% confidence interval given a stdev and number of pairs
    Args:
        stdev_gamma: the standard deviation of the bin
        num_pairs: the number of pair in the bin

    Returns: magnitude of confidence interval
    """
    ci = 1.96 * stdev_gamma / math.sqrt(num_pairs)
    return ci


def down_sample(data: list, factor: int) -> list:
    """
    Helper function for down-sampling a list
    Args:
        data: the data to sample from
        factor: the factor to down-sample by

    Returns: list of down sampled data
    """
    down_sampled = []
    for i, d in enumerate(data):
        if i % factor == 0:
            down_sampled.append(d)
    return down_sampled