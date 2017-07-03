"""
@author: Nikhil Bhaip
@version: 2.0
@since: 6/13/16

The S&P 500 program implements an algorithm that finds the percentage change in the S&P 500 Index based on historical
weekly data and visualizes the information as a time series plot in matplotlib. The program also creates a Markov chain
model in which the states are bull market, bear market, and stagnant market. Using the probabilities associated with
this Markov chain model, the program will predict the future S&P 500 data through a random walk.

The next step would be to change other variables like periodicity (daily, weekly, monthly, etc.), use stock data rather
than S&P 500, and incorporate other newer variables like seasonality.

"""
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter
from matplotlib.ticker import MultipleLocator
import matplotlib.mlab as mlab
import numpy as np


def get_data():
    """
    This function obtains weekly S&P500 data from the last 7 years as a DataFrame from Quandl. I'm mainly interested
    in the date, adjusted_closing and the percentage change in adj_closing from the last week.

    :return: A Pandas DataFrame with columns: Date, Adjusted Close, and Percentage Change.
    """
    sp_500_df = pd.read_csv("https://www.quandl.com/api/v3/datasets/YAHOO/INDEX_GSPC.csv?collapse=weekly" +
                            "&start_date=2009-05-23&end_date=2016-05-23&api_key=7NU4-sXfczxA9fsf_C8E")
    adjusted_df = sp_500_df.ix[:, ['Date', 'Adjusted Close']]
    adjusted_df["Percentage Change"] = adjusted_df['Adjusted Close'][::-1].pct_change() * 100
    print(adjusted_df)
    return adjusted_df
#get_data()


def percent_change_as_time_plot(adjusted_df):
    """
    This function visualizes the percentage change data as a time series plot.

    :param adjusted_df: Pandas DataFrame with columns: Date, Adjusted Close, and Percentage Change.
    """

    pct_change_list = adjusted_df['Percentage Change'].tolist()
    date_list = [dt.datetime.strptime(d, '%Y-%m-%d').date() for d in adjusted_df['Date'].tolist()]
    fig, ax = plt.subplots()
    ax.plot(date_list, pct_change_list)
    plt.xlabel("Years")
    plt.ylabel("Percentage change from last week")
    plt.title("Percentage change in S&P 500 weekly data from 2009 to 2016")
    ax.xaxis.set_minor_locator(MonthLocator())
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.fmt_xdata = DateFormatter('%Y-%m-%d')
    ax.autoscale_view()
    fig.autofmt_xdate()

    plt.show()


def get_params_for_norm_dist(adjusted_df):
    """
    This function returns the mean and standard deviation in the percentage change column of a DataFrame.
    """
    mean = adjusted_df["Percentage Change"].mean()
    std = adjusted_df["Percentage Change"].std()
    return mean, std


def percent_change_as_hist(adjusted_df):
    """
    This function visualizes the percentage change data as a histogram. The graph is also fitted to a normal
     distribution curve.

    :param adjusted_df: Pandas DataFrame with columns: Date, Adjusted Close, and Percentage Change.
    """
    pct_change_list = adjusted_df['Percentage Change']

    # Code below removes the NaN value and plots the histogram. Bins are left adjusted right now, so when plotting the
    # normal distribution function, we must adjust it to be based off the center (average) of the bins.
    n, bins, patches = plt.hist(pct_change_list.dropna(), bins=25, normed=True)
    bincenters = 0.5*(bins[1:]+bins[:-1])

    plt.xlabel("Percentage change")
    plt.ylabel("Frequency")
    mean, std = get_params_for_norm_dist(adjusted_df)
    plt.title("Distribution of percentage change in S&P 500. Mu: %.3f, Sigma: %.3f" % (mean, std), y=1.03)

    # adds vertical lines to the graph corresponding to the x's that represent the number of deviations from the mean
    for num_std_from_mean in range(-3, 4):
        plt.axvline(mean + std * num_std_from_mean)

    # plots the normal pdf of best fit
    y = mlab.normpdf(bincenters, mean, std)
    plt.plot(bincenters, y, 'r--', linewidth=1)

    plt.show()


def percent_change_prob(adjusted_df, threshold=0):
    """
    This function finds the probabilities associated with the Markov chain of the percentage change column. There are
    two states: % change greater than or equal to a threshold (A) and % change less than a threshold (B). The threshold
    is defaulted to zero, so that the states are roughly divided into positive and negative changes. The four
    probabilities are: a P(A | A), b (A | B) , c (B | A) , d (B | B). The sum of the rows in the matrix must add up to
    1: (a + b = 1 and c + d = 1)

            A   B
    P = A   a   b
        B   c   d

    :param adjusted_df: <DataFrame> Pandas DataFrame with columns: Date, Adjusted Close, and Percentage Change.
    :param threshold: <float> Represents the level dividing events A (change >= threshold) & B (change < threshold).

    """

    a_count = 0  # counts frequency of when A occurs then the next week A occurs
    b_count = 0  # counts frequency of when A occurs then the next week B occurs
    c_count = 0  # counts frequency of when B occurs then the next week A occurs
    d_count = 0  # counts frequency of when B occurs then the next week B occurs

    # returns a series of % change without missing data and reverses the order of data, so it starts from earliest date
    pct_change_series = adjusted_df['Percentage Change'].dropna().iloc[::-1]
    print(pct_change_series)
    for index, pct in pct_change_series.iteritems():
        if index == 0:  # prevents program from calculating a future probability
            continue
        if pct >= threshold:
            if pct_change_series[index-1] >= threshold:
                a_count += 1
            else:
                b_count += 1
        else:
            if pct_change_series[index-1] >= threshold:
                c_count += 1
            else:
                d_count += 1
        print(index)

    # Given event A just happened, this is the probability that another event A occurs
    a_prob = a_count / (a_count + b_count)

    # Given event A just happened, this is the probability that event B occurs
    b_prob = b_count / (a_count + b_count)

    # Given event B just happened, this is the probability that event A occurs
    c_prob = c_count / (c_count + d_count)

    # Given event B just happened, this is the probability that another event B occurs
    d_prob = d_count / (c_count + d_count)

    prob_list = [[a_prob, b_prob], [c_prob, d_prob]]
    print(prob_list, "\n")

    print("\nIf the S&P 500 has a positive percentage change this week, there is a %.3f chance that "
          "next week there will be a repeat positive percentage change. If the index rises this week, then there is a "
          "%.3f chance that next week the index will fall. However, if the S&P drops in one week there is a %.3f that"
          " next week there will be a repeat negative percentage change. If the index falls this week, then there is a "
          "%.3f chance that the index will rise next week. \n" % (a_prob, b_prob, d_prob, c_prob))
    return prob_list


def random_walk_norm_pdf(adjusted_df, start=2099, num_periods=12):
    """
    This function calculates and visualizes a random walk assuming that S&P 500 data are independent of current state.
    Based on a basic normal distribution and a starting point, the function will predict the S&P 500
    Index movement for a finite number of periods. This is the most fundamental random walk and has many unrealistic
    assumptions, such as the data are independently and identically distributed, which is likely not true for the
    S&P500 Index.

    :param adjusted_df: <DataFrame> Pandas DataFrame with columns: Date, Adjusted Close, and Percentage Change.
    :param start: <float> starting value for S&P 500 random walk
    :param num_periods: <int> number of steps in the random walk process

    """
    mean, std = get_params_for_norm_dist(adjusted_df)
    pct_change_list = []
    all_walks = []  # will hold all the random walk data
    for i in range(100):
        random_walk = [start]
        for period in range(num_periods):
            # sets the step as the last element in the random walk
            step = random_walk[-1]

            # picks a random percent change from a Gaussian distribution based on historical mean and standard deviation
            pct_change = np.random.normal(mean, std)
            pct_change_list.append(pct_change)

            # reordering of percent change formula
            step = ((pct_change * step / 100) + step)

            random_walk.append(step)
        all_walks.append(random_walk)
    show_rand_walks(all_walks)


def prob_from_bins(heights, bins):
    """
    Chooses a random bin based on the prob distribution in the histogram. Then returns a random percentage change from
    that bin.

    :param heights: <list> heights of the histogram
    :param bins: <list> left-hand edges of each bin; must have at least two values in list
    :return: <float> random percentage change
    """
    np_heights = np.asarray(heights)
    bin_length = bins[1]-bins[0]
    np_area = bin_length * np_heights  # sum of area is equal to 1
    bin_num = np.random.choice(np.arange(start=1, stop=len(bins)), p=np_area)
    rand_pct_change = bin_length * np.random.ranf() + bins[bin_num-1]
    return rand_pct_change


def rand_walk_2x2_markov(adjusted_df, prob_list, num_bins=10, threshold=0, start=2099, num_periods=12):
    """
    Divides the per

    :param adjusted_df: <DataFrame> Pandas DataFrame with columns: Date, Adjusted Close, and Percentage Change.
    :param prob_list: <list> Contains a 2x2 list that holds the probabilities from a Markov chain with two states
    :param num_bins: <int> Specifies number of bins in the histogram distribution. The more bins, the more realistic
                            the probability distribution will be
    :param threshold: <float> Represents the level dividing events A (change >= threshold) & B (change < threshold)
    :param start: <float> starting value for S&P 500 random walk
    :param num_periods: <int> number of steps in the random walk process
    """

    pct_change_array = np.array(adjusted_df["Percentage Change"].dropna())
    pct_above_threshold_array = pct_change_array[pct_change_array >= threshold]
    pct_below_threshold_array = pct_change_array[pct_change_array < threshold]
    n_above, bins_above, patches_above = plt.hist(pct_above_threshold_array, bins=num_bins, normed=True)
    n_below, bins_below, patches_below = plt.hist(pct_below_threshold_array, bins=num_bins, normed=True)

    # First percentage change is determined from a basic normal distribution. Every following period is based on the
    # percentage change of the previous period
    pct_change_list = []
    all_walks = []  # will hold all the random walk data
    for i in range(100):
        mean, std = get_params_for_norm_dist(adjusted_df)
        first_pct_change = np.random.normal(mean, std)
        pct_change_list.append(first_pct_change)
        first_step = ((first_pct_change * start / 100) + start)
        random_walk = [start, first_step]

        for period in range(num_periods):
            step = random_walk[-1]
            prev_pct_change = pct_change_list[-1]

            # random number used to test whether event A will occur or event B will occur
            rand_prob = np.random.random_sample()
            if prev_pct_change >= threshold:  # If true, event A occurred
                # prob_list[0][0] is probability that another event A will occur, given event A has happened already
                if rand_prob <= prob_list[0][0]:  # If true, A then A
                    pct_change = prob_from_bins(n_above, bins_above)
                else:  # If true, A then B
                    pct_change = prob_from_bins(n_below, bins_below)
            else:  # If true, event B occurred
                # prob_list[1][0] is probability that event A will occur, given event B has happened already
                if rand_prob <= prob_list[1][0]:  # If true, B then A
                    pct_change = prob_from_bins(n_above, bins_above)
                else:  # If true, B then B
                    pct_change = prob_from_bins(n_below, bins_below)

            pct_change_list.append(pct_change)

            step = ((pct_change * step / 100) + step)

            random_walk.append(step)
        all_walks.append(random_walk)
    show_rand_walks(all_walks)


def show_rand_walks(all_walks):
    """
    Visualizes all random walks as a plot and distribution.

    :param all_walks: list of all random walks conducted
    """
    np_aw = np.array(all_walks)  # converts the list of all random walks to a Numpy Array
    np_aw_t = np.transpose(np_aw)  # must transpose the array for graph to display properly
    plt.clf()
    plt.plot(np_aw_t)
    plt.xlabel("Steps")
    plt.ylabel("S&P 500 Index Value")
    plt.title("All Random Walks of the S&P 500 Index")
    plt.show()

    # Select last row from np_aw_t: ends
    ends = np_aw_t[-1]

    # Plot histogram of ends, display plot
    n, bins, patches = plt.hist(ends, bins=25, normed=True)
    plt.xlabel("Final S&P 500 Index Value at end of period.")
    plt.ylabel("Frequency")
    rand_mean = ends.mean()
    rand_std = ends.std()

    plt.title("Distribution of Random Walk Final Values. Mean is %d and Standard Deviation is %d"
              % (rand_mean, rand_std), y=1.03)
    for num_std_from_mean in range(-3, 4):
        plt.axvline(rand_mean + rand_std * num_std_from_mean)
    bincenters = 0.5*(bins[1:]+bins[:-1])
    # plots the normal pdf of best fit
    y = mlab.normpdf(bincenters, rand_mean, rand_std)
    plt.plot(bincenters, y, 'r--', linewidth=3)
    plt.show()


my_sp_500_df = get_data()
#print(my_sp_500_df)
#percent_change_as_time_plot(my_sp_500_df)
#percent_change_as_hist(my_sp_500_df)
prob_matrix = percent_change_prob(my_sp_500_df)
#random_walk_norm_pdf(my_sp_500_df, num_periods=12)
#rand_walk_2x2_markov(my_sp_500_df, prob_list=prob_matrix, threshold=0)