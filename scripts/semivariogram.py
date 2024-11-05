import argparse
import itertools
import matplotlib.pyplot as plt
import utils


def raw_semivariogram(xs: list[float],
                      ys: list[float],
                      variable: list[float],
                      variable_name: str,
                      zs: list[float] = None) -> tuple[list, list]:
    """
    Create a raw semivariogram plot.
    Args:
        xs: list of x locations for the variable
        ys: list of y locations for the variable
        variable: list of values for the variable
        variable_name: name of the variable
        zs: (optional) list of z locations for the variable

    Returns: (tuple) ranges between all variable pairs, semivariance for all variable pairs
    """
    if len(xs) != len(ys) or len(xs) != len(variable) or (zs is not None and len(xs) != len(zs)):
        raise Exception(f"Must have the same number of x values, y, values, z values (if specified) and variable "
                        f"values. Found: {len(xs)}, {len(ys)}, {len(zs) if zs is not None else 'NA'}, {len(variable)}")

    pairs = list(itertools.combinations(range(len(xs)), 2))
    print(f"Num pairs: {len(pairs)}")

    # okay let's store ranges and variances in the same data structure so that we can easily sort by range later.
    # how about a list of tuples...
    r_gamma_pairs = []
    for p_near, p_far in pairs:

        # make range calculation for pair
        x1 = xs[p_near]
        x2 = xs[p_far]
        y1 = ys[p_near]
        y2 = ys[p_far]
        z1 = zs[p_near] if zs is not None else 0.0
        z2 = zs[p_far] if zs is not None else 0.0
        r = utils.calculate_range(x1, x2, y1, y2, z1, z2)

        # calculate semivariance for pair
        v_near = variable[p_near]
        v_far = variable[p_far]
        gamma = utils.semivariance_pair(v_near, v_far)
        r_gamma_pairs.append((r, gamma))

    # now sort the pairs in preparation for binning later
    sorted_pairs = sorted(r_gamma_pairs, key=lambda x: x[0])
    ranges = [x[0] for x in sorted_pairs]
    gammas = [x[1] for x in sorted_pairs]

    # make a plot of the raw semivariances
    plt.clf()
    plt.scatter(ranges, gammas, color="gray")
    plt.ylabel("Semivariance")
    plt.xlabel("Distance (m)")
    plt.title(f"Raw Semivariogram for {variable_name}")
    plt.savefig(f"raw_semivariogram_{variable_name}.png")
    return ranges, gammas


def semivariogram(ranges: list, gammas: list, bin_cutoffs: list, variable_name: str) -> tuple[list, list, list, list]:
    """
    Create a semivariogram plot for a dataset
    Args:
        ranges: list of range values for each pair in the data
        gammas: list of gamma values for each pair in the data
        bin_cutoffs: list of ranges to cut the bins off at. Ex: [1, 3, 5, 10, 20]
        variable_name: name of the variable

    Returns: list of mean range values for each bin
             list of mean semivariance values for each bin
             list of the standard deviation values for each bin
             list of the number of values in each bin
    """
    range_means, gamma_means, stdev_gammas, bin_counts = utils.create_mean_bins(ranges, gammas, bin_cutoffs=bin_cutoffs)

    # plot the raw first so that its on the bottom, then plot the semivariogram on top
    plt.clf()
    plt.scatter(ranges, gammas, color="gray", label="raw")
    plt.scatter(range_means, gamma_means, color="black", label="means")
    plt.title(f"Semivariogram for {variable_name}")
    plt.ylabel("Semivariance")
    plt.xlabel("Distance (m)")
    plt.legend(loc='upper right')
    plt.savefig(f"semivariogram_{variable_name}_with_raw.png")

    # then plot semivariogram on its own
    plt.clf()
    plt.scatter(range_means, gamma_means, color="black", label="means")
    plt.title(f"Semivariogram for {variable_name}")
    plt.ylabel("Semivariance")
    plt.xlabel("Distance (m)")
    plt.savefig(f"semivariogram_{variable_name}.png")
    return range_means, gamma_means, stdev_gammas, bin_counts


def fit_semivariogram(ranges: list,
                      gammas: list,
                      stdev_gammas: list,
                      bin_counts: list,
                      fit_type: int,
                      w: float,
                      a: float,
                      variable_name: str) -> None:
    """
    Fit a model to a semivariogram
    Args:
        ranges: list of mean range values for each bin
        gammas: list of mean gamma values for each bin
        stdev_gammas: list of standard deviation values for each bin
        bin_counts: list of the number of values in each bin
        fit_type: the type of model to fit to the 1 for exponential, 2 for spherical, 3 for gaussian
        w: the sill of the model
        a: the range of the model
        variable_name: the name of the variable
    """
    fit_map = {
        1: "exponential",
        2: "spherical",
        3: "gaussian"
    }
    try:
        fit_choice = fit_map[fit_type]
    except KeyError as e:
        print(f"Unknown fit type: {fit}. Allowed options are: 1, 2, 3")
        raise e
    else:
        fit_x, fit_y = getattr(utils, fit_choice)(max_range=int(max(ranges)), w=w, a=a)
        plt.clf()

        # plot the range and the sill first so that they are on the "bottom"
        plt.plot([0, max(ranges)], [w, w], label="Sill", color="blue")
        plt.plot([a, a], [0, max(gammas)], label="range", color="purple")

        # then plot the data
        plt.scatter(ranges, gammas, color="gray")

        # plot the CI
        y_lower, y_upper = utils.confidence_interval_model(gammas, stdev_gammas, bin_counts)
        plt.plot(ranges, y_lower, color="gray", label="95% CI")
        plt.plot(ranges, y_upper, color="gray")

        # finally plot the fit
        plt.plot(fit_x, fit_y, color="black")
        plt.legend()
        plt.ylabel("Semivariance")
        plt.xlabel("Distance (m)")
        plt.title(f"Semivariogram with {fit_choice} fit for {variable_name}")
        plt.savefig(f"semivariogram_{variable_name}_with_fit.png")


if __name__ == "__main__":

    # in general, this should be run 3 times by passing the folling parameter:
    # 1. just provide the --data arg on its own, then inspect and put together a set of ranges in a .txt
    # file according to the raw semivariance plot that is generated
    # 2. pass the --semivariogram arg with a set of bin cut offs in a txt file to create a semivariogram
    # 3. pass the --fit arg to generate a fit

    # once happy with the results, the whole suite can be run together by passing all necessary args at the same time.
    parser = argparse.ArgumentParser(description="Script to create semivariogram plots for a csv formatted dataset.")
    parser.add_argument("--data",
                        help="Path to dataset",
                        required=True)
    parser.add_argument("--x_name",
                        help="Name of the x variable in the dataset",
                        required=False,
                        default="x")
    parser.add_argument("--y_name",
                        help="Name of the y variable in the dataset",
                        required=False,
                        default='y')
    parser.add_argument("--z_name",
                        help="Name of the z variable in the dataset",
                        required=False,
                        default=None)
    parser.add_argument("--v_name",
                        help="The primary variable to create variograms for",
                        required=True)
    parser.add_argument("--cutoffs_file",
                        help="Path to a .txt file containing a list of rang cutoffs",
                        required=False,
                        default=None)
    parser.add_argument("--fit",
                        help="Input a list of 4 comma separated values, where the first value specifies the fit "
                             "(0 for no fit, 1 for exponential fit, 2 for spherical fit, 3 for Gaussian fit)",
                        required=False,
                        default="0")
    parser.add_argument("--sill",
                        help="Sill value for a fit",
                        required=False,
                        default="0")
    parser.add_argument("--range",
                        help="Range value for a fit",
                        required=False,
                        default="0")
    parser.add_argument("--down_sample_factor",
                        help="Factor to down-sample by",
                        required=False,
                        default="1")
    args = parser.parse_args()

    dataset = utils.load_csv_dataset(args.data)

    # down-sample the data, if specified. Sometimes this is needed for large datasets.
    down_sample_factor = int(args.down_sample_factor)

    # get all relevant values from the dataset
    x_values = utils.down_sample(list(dataset.get(args.x_name)), down_sample_factor)
    y_values = utils.down_sample(list(dataset.get(args.y_name)), down_sample_factor)
    z_values = utils.down_sample(list(dataset.get(args.z_name)), down_sample_factor) if args.z_name is not None else None
    primary_values = utils.down_sample(list(dataset.get(args.v_name)), down_sample_factor)

    rs, gs = raw_semivariogram(x_values, y_values, primary_values, args.v_name, z_values)

    if args.cutoffs_file is not None:
        bin_cuts = utils.load_bins_from_file(args.cutoffs_file)
        r_means, g_means, std_means, b_counts = semivariogram(rs, gs, bin_cuts, args.v_name)

        fit = int(args.fit)
        if bool(fit):
            # then let's fit stuff
            fit_semivariogram(r_means,
                              g_means,
                              std_means,
                              b_counts,
                              fit,
                              w=float(args.sill),
                              a=float(args.range),
                              variable_name=args.v_name)
