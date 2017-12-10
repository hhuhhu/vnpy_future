"""
@author: Nikhil Bhaip
@version: 2.4
@since: 6/20/16

The markov_stock analysis program implements an algorithm that finds the percentage change in a security based on
historical weekly data from Yahoo Finance and visualizes the information as a time series plot in matplotlib. The
program also creates a Markov chain model in which the states are bull market, bear market, and stagnant market.
Using the probabilities associated with this Markov chain model, the program will predict the future S&P 500 data
through a random walk. This program can be used as a tool to analyze securities, like stocks and indexes, as well as
study the state of the market for a wide number of applications including options and technical analysis.

The next step would be to include other newer variables like seasonality.

"""
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter
from matplotlib.ticker import MultipleLocator
import matplotlib.mlab as mlab
import pandas as pd
import numpy as np
import quandl
quandl.ApiConfig.api_key = "7NU4-sXfczxA9fsf_C8E"


class SecurityInfo:
    """
    Holds information about a security (stock, index) to be used when retrieving data from Quandl and accessing
    information for other functions within the program.

    """
    def __init__(self, name, start, end, period="weekly"):
        self.name = name
        try:
            dt.datetime.strptime(start, '%Y-%m-%d')
            dt.datetime.strptime(end, '%Y-%m-%d')
            self.start = start
            self.end = end
        except ValueError:
            raise ValueError("Incorrect data format, should be YYYY-MM-DD")
        period_list = ["none", "daily", "weekly", "monthly", "quarterly", "annual"]
        if period in period_list:
            self.period = period
        else:
            print("Invalid period format. Using default period as 'weekly'")
            self.period = "weekly"

    def summary(self):
        print("Name: " + self.name)
        print("Starting Date: " + self.start)
        print("Ending Date: " + self.end)
        print("Period: " + self.period)

    def valid_date(self, new_date):
        try:
            dt.datetime.strptime(new_date, '%Y-%m-%d')
            return True
        except ValueError:
            raise ValueError("Incorrect data format, should be YYYY-MM-DD")

    def set_name(self, new_name):
        self.name = new_name

    def get_name(self):
        return self.name

    def set_start(self, new_start):
        if self.valid_date(new_start):
            self.start = new_start

    def get_start(self):
        return self.start

    def set_end(self, new_end):
        if self.valid_date(new_end):
            self.end = new_end

    def get_end(self):
        return self.end

    def set_period(self, new_period):
        self.period = new_period

    def get_period(self):
        return self.period


def get_data(security):
    """
    This function obtains data under certain parameters from Quandl and returns the following information as a Pandas
    DataFrame: date, adjusted closing, and percentage change in adjusted closing from the last week.

    :param security: <SecurityInfo class> Holds information about the requested security
    :return: A Pandas DataFrame with columns: Date, Adjusted Close, and Percentage Change.
    """
    name = security.get_name()
    start = security.get_start()
    end = security.get_end()
    period = security.get_period()
    raw_df = quandl.get("YAHOO/" + name, start_date=start, collapse=period)
    adjusted_df = raw_df.ix[:, ['Adjusted Close']]
    adjusted_df["Percentage Change"] = adjusted_df['Adjusted Close'].pct_change() * 100
    return adjusted_df


def percent_change_as_time_plot(adjusted_df, security):
    """
    This function visualizes the percentage change data as a time series plot.

    :param adjusted_df: Pandas DataFrame with columns: Date, Adjusted Close, and Percentage Change.
    :param security: <SecurityInfo class> Holds information about the requested security
    """

    pct_change_list = adjusted_df['Percentage Change'].tolist()
    date_list = adjusted_df.index.values
    fig, ax = plt.subplots()
    ax.plot(date_list, pct_change_list)
    plt.xlabel("Dates")
    plt.ylabel("Percentage change from last period")
    if security.get_period() == "none":
        plt.title("Percentage change in " + security.get_name(), y=1.03)
    else:
        plt.title("Percentage change in " + security.get_name() + " " + security.get_period() + " data", y=1.03)
    ax.xaxis.set_minor_locator(MonthLocator())
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.fmt_xdata = DateFormatter('%Y-%m-%d')
    ax.autoscale_view()
    fig.autofmt_xdate()

    plt.show()


def get_params_for_norm_dist(adjusted_df):
    """
    This function returns the mean and standard deviation in the percentage change column of a DataFrame.
    :param adjusted_df: <Data Frame> must have 'Percentage Change' column

    :returns mean and standard deviation of the percentage change column
    """
    mean = adjusted_df["Percentage Change"].mean()
    std = adjusted_df["Percentage Change"].std()
    return mean, std


def percent_change_as_hist(adjusted_df, security):
    """
    This function visualizes the percentage change data as a histogram. The graph is also fitted to a normal
     distribution curve.

    :param security: <SecurityInfo class> Holds information about the requested security
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
    plt.title("Distribution of percentage change in " + security.get_name() + " Mu: %.3f, Sigma: %.3f"
              % (mean, std), y=1.03)

    # adds vertical lines to the graph corresponding to the x's that represent the number of deviations from the mean
    for num_std_from_mean in range(-3, 4):
        plt.axvline(mean + std * num_std_from_mean)

    # plots the normal pdf of best fit
    y = mlab.normpdf(bincenters, mean, std)
    plt.plot(bincenters, y, 'r--', linewidth=1)

    plt.show()


def print_prob_list_2x2(security, threshold, prob_list):
    """
    Outputs summary of a 2x2 probability matrix

    :param security: <SecurityInfo class> Holds information about the requested security
    :param threshold: <float> Represents the level dividing events A (change >= threshold) & B (change < threshold).
    :param prob_list: <list> 2x2 markov transition matrix describing security
    """
    print(prob_list, "\n")
    thresh_str = "{:.2f}%".format(threshold)
    for i in range(2):
        if i == 0:
            part_1_summary = "\nIf " + security.get_name() + " rises above {thresh} in one period, "
        else:
            part_1_summary = "\nIf " + security.get_name() + " falls below {thresh} in one period, "
        part_2_summary = "in the next period, there is a {:.2f} chance that the security will rise above {thresh} " \
                         "and a {:.2f} chance that it will fall below this threshold."
        print((part_1_summary + part_2_summary).format(prob_list[i][0], prob_list[i][1],
                                                       thresh=thresh_str))


def percent_change_prob_2x2(adjusted_df, security, threshold=0.0, summary=False):
    """
    This function finds the probabilities associated with the Markov chain of the percentage change column. There are
    two states: % change greater than or equal to a threshold (A) and % change less than a threshold (B). The threshold
    is defaulted to zero, so that the states are roughly divided into positive and negative changes. The four
    probabilities are: a (A | A), b (B | A) , c (A | B) , d (B | B). By definition, the sum of the rows in the right
    stochastic transition matrix must add up to 1: (a + b = 1 and c + d = 1)

            A   B
    P = A   a   b
        B   c   d

    :param adjusted_df: <DataFrame> Pandas DataFrame with columns: Date, Adjusted Close, and Percentage Change.
    :param security: <SecurityInfo class> Holds information about the requested security
    :param threshold: <float> Represents the level dividing events A (change >= threshold) & B (change < threshold).
    :param summary: <Boolean> If true, outputs a summary of the probability matrix
    """
    count_list = [[0, 0],    # a_count, b_count,
                  [0, 0]]    # c_count, d_count

    new_df = adjusted_df['Percentage Change'].dropna().to_frame()
    new_df = new_df.set_index(np.arange(new_df.size, 0, -1))

    for index, pct in new_df['Percentage Change'].iteritems():
        if index == 1:  # prevents program from calculating a future probability
            break
        if pct >= threshold:
            if new_df['Percentage Change'][index-1] >= threshold:
                count_list[0][0] += 1  # event A occurred, then event A occurred
            else:
                count_list[0][1] += 1  # event A occurred, then event B occurred
        else:
            if new_df['Percentage Change'][index-1] >= threshold:
                count_list[1][0] += 1  # event B occurred, then event A occurred
            else:
                count_list[1][1] += 1  # event B occurred, then event B occurred

    prob_list = [[count / sum(group) for count in group] for group in count_list]

    if summary:
        print_prob_list_2x2(security, threshold, prob_list)

    return prob_list


def percent_change_prob_3x3(adjusted_df, security, lower_thresh=-1.0, upper_thresh=1.0):
    """
    This function finds the probabilities associated with the Markov chain of the percentage change column. There are
    three states: % change less than or equal to a lower threshold (A), % change between the upper and lower
    thresholds (B)and % change greater than or equal to an upper threshold (C). The lower threshold is defaulted to -1,
    and the upper threshold is defaulted to +1. Percentage changes below the lower threshold may be considered bearish,
    in between the two thresholds considered stagnant, and above the threshold considered bullish. The nine
    probabilities are: a P(A | A), b (B | A) , c (C | A) , d (A | B), e (B | B), f (C | B), g (A | C), h (B | C), and
    i (C | C). The sum of the rows in the matrix must add up to 1: (a + b + c = 1 and d + e + f = 1 and g + h + i = 1)

            A   B   C
    P = A   a   b   c
        B   d   e   f
        C   g   h   i

    See percent_change_prob_2x2 for more details

    :param adjusted_df: <DataFrame> Pandas DataFrame with columns: Date, Adjusted Close, and Percentage Change.
    :param security: <SecurityInfo class> Holds information about the requested security
    :param lower_thresh: <float> Represents the level dividing events A & B
    :param upper_thresh: <float> Represents the level dividing events A & B
    """
    # counts frequency of sub-events
    count_list = [[0, 0, 0],    # a_count, b_count, c_count
                  [0, 0, 0],    # d_count, e_count, f_count
                  [0, 0, 0]]    # g_count, h_count, i_count

    new_df = adjusted_df['Percentage Change'].dropna().to_frame()
    new_df = new_df.set_index(np.arange(new_df.size, 0, -1))

    for index, pct in new_df['Percentage Change'].iteritems():
        if index == 1:  # prevents program from calculating a future probability
            break
        if pct <= lower_thresh:

            if new_df['Percentage Change'][index-1] <= lower_thresh:
                count_list[0][0] += 1  # increment a_count
            elif lower_thresh < new_df['Percentage Change'][index-1] < upper_thresh:
                count_list[0][1] += 1  # increment b_count
            else:
                count_list[0][2] += 1  # increment c_count

        elif lower_thresh < pct < upper_thresh:

            if new_df['Percentage Change'][index-1] <= lower_thresh:
                count_list[1][0] += 1  # increment d_count
            elif lower_thresh < new_df['Percentage Change'][index-1] < upper_thresh:
                count_list[1][1] += 1  # increment e_count
            else:
                count_list[1][2] += 1  # increment f_count

        else:

            if new_df['Percentage Change'][index-1] <= lower_thresh:
                count_list[2][0] += 1  # increment g_count
            elif lower_thresh < new_df['Percentage Change'][index-1] < upper_thresh:
                count_list[2][1] += 1  # increment h_count
            else:
                count_list[2][2] += 1  # increment i_count

    prob_list = [[count / sum(group) for count in group] for group in count_list]
    for group in prob_list:
        print(group)
    lower_thresh_str = "{:.2f}%".format(lower_thresh)
    upper_thresh_str = "{:.2f}%".format(upper_thresh)
    for i in range(3):
        if i == 0:
            part_1_summary = "\nIf " + security.get_name() + " falls below {lower_thresh} in one period (bearish),"
        elif i == 1:
            part_1_summary = "\nIf " + security.get_name() + " changes between {lower_thresh} and {upper_thresh} in " \
                                                             "one period (stagnant),"
        else:
            part_1_summary = "\nIf " + security.get_name() + " rises above {upper_thresh} in one period (bullish),"

        part_2_summary = "in the next period, there is a {:.3f} chance that the security will fall by more than " \
                         "{lower_thresh} (bearish), a {:.3f} chance that the security will change somewhere between " \
                         "{lower_thresh} and {upper_thresh} (stagnant), and a {:.3f} chance that the security will " \
                         "rise by more than {upper_thresh} (bullish)."
        print((part_1_summary + part_2_summary).format(prob_list[i][0], prob_list[i][1], prob_list[i][2],
                                                       lower_thresh=lower_thresh_str, upper_thresh=upper_thresh_str))

    return prob_list


def random_walk_norm_pdf(adjusted_df, security, start, num_periods=12):
    """
    This function calculates and visualizes a random walk assuming that security data are independent of current state.
    Based on a basic normal distribution and a starting point, the function will predict the security's movement for a
    finite number of periods. This is the most fundamental random walk and has many unrealistic
    assumptions, such as the data are independently and identically distributed, which is likely not true for the
    most securities.

    :param adjusted_df: <DataFrame> Pandas DataFrame with columns: Date, Adjusted Close, and Percentage Change.
    :param security: Holds information about a selected security
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
    show_rand_walks(all_walks, security)


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


def rand_walk_2x2_markov(adjusted_df, prob_list, security, num_bins=10, threshold=0.0, start=2099.0, num_periods=12):
    """
    Divides the per

    :param adjusted_df: <DataFrame> Pandas DataFrame with columns: Date, Adjusted Close, and Percentage Change.
    :param prob_list: <list> Contains a 2x2 list that holds the probabilities from a Markov chain with two states
    :param security: <SecurityInfo class> Holds information about the requested security
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
    show_rand_walks(all_walks, security)


def rand_walk_3x3_markov(adjusted_df, prob_list, security, num_bins=10, lower_thresh=-1.0, upper_thresh=1.0,
                         start=2099.0, num_periods=12):
    """

    :param adjusted_df: <DataFrame> Pandas DataFrame with columns: Date, Adjusted Close, and Percentage Change.
    :param prob_list: <list> Contains a 2x2 list that holds the probabilities from a Markov chain with two states
    :param security: <SecurityInfo class> Holds information about the requested security
    :param num_bins: <int> Specifies number of bins in the histogram distribution. The more bins, the more realistic
                            the probability distribution will be
    :param lower_thresh: <float> Represents the level dividing events A (pct change < lower thresh) & B(lower thresh <=
    pct change < upper thresh)
    :param upper_thresh: <float> Represents the level dividing events B (lower thresh < pct change < upper thresh) &
    C(upper thresh < pct change)
    :param start: <float> starting value for S&P 500 random walk
    :param num_periods: <int> number of steps in the random walk process
    """

    pct_change_array = np.array(adjusted_df["Percentage Change"].dropna())
    pct_above_array = pct_change_array[pct_change_array >= upper_thresh]
    pct_between_array = pct_change_array[np.logical_and(pct_change_array > lower_thresh,
                                                        pct_change_array < upper_thresh)]
    pct_below_array = pct_change_array[pct_change_array <= lower_thresh]
    n_above, bins_above, patches_above = plt.hist(pct_above_array, bins=num_bins, normed=True)
    n_between, bins_between, patches_between = plt.hist(pct_between_array, bins=num_bins, normed=True)
    n_below, bins_below, patches_below = plt.hist(pct_below_array, bins=num_bins, normed=True)

    # First percentage change is determined from a basic normal distribution. Every following period is based on the
    # percentage change of the previous period
    pct_change_list = []
    all_walks = []  # will hold all the random walk data
    for i in range(1000):
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
            if prev_pct_change <= lower_thresh:  # If true, event A occurred
                # prob_list[0][0] is probability that another event A will occur, given event A has happened already
                if 0 < rand_prob <= prob_list[0][0]:  # If true, A then A
                    pct_change = prob_from_bins(n_below, bins_below)
                elif prob_list[0][0] < rand_prob < (prob_list[0][0] + prob_list[0][1]):  # If true, A then B
                    pct_change = prob_from_bins(n_between, bins_between)
                else:  # If true, A then C
                    pct_change = prob_from_bins(n_above, bins_above)

            elif lower_thresh < prev_pct_change < upper_thresh:  # If true, event B occurred
                # prob_list[1][0] is probability that  event A will occur, given event B has happened already
                if 0 < rand_prob <= prob_list[1][0]:  # If true, B then A
                    pct_change = prob_from_bins(n_below, bins_below)
                elif prob_list[1][0] < rand_prob < (prob_list[1][0] + prob_list[1][1]):  # If true, B then B
                    pct_change = prob_from_bins(n_between, bins_between)
                else:  # If true, B then C
                    pct_change = prob_from_bins(n_above, bins_above)

            else:  # If true, event C occurred
                # prob_list[2][0] is probability that  event A will occur, given event C has happened already
                if 0 < rand_prob <= prob_list[2][0]:  # If true, C then A
                    pct_change = prob_from_bins(n_below, bins_below)
                elif prob_list[2][0] < rand_prob < (prob_list[2][0] + prob_list[2][1]):  # If true, C then B
                    pct_change = prob_from_bins(n_between, bins_between)
                else:  # If true, C then C
                    pct_change = prob_from_bins(n_above, bins_above)

            pct_change_list.append(pct_change)

            step = ((pct_change * step / 100) + step)

            random_walk.append(step)
        all_walks.append(random_walk)
    show_rand_walks(all_walks, security)


def show_rand_walks(all_walks, security):
    """
    Visualizes all random walks as a plot and distribution.

    :param all_walks: list of all random walks conducted
    :param security: holds information about the security
    """
    np_aw = np.array(all_walks)  # converts the list of all random walks to a Numpy Array
    np_aw_t = np.transpose(np_aw)  # must transpose the array for graph to display properly
    plt.clf()
    plt.plot(np_aw_t)
    plt.xlabel("Steps")
    plt.ylabel("Value of " + security.get_name())
    plt.title("All Random Walks of " + security.get_name())
    plt.show()

    # Select last row from np_aw_t: ends
    ends = np_aw_t[-1]

    # Plot histogram of ends, display plot
    n, bins, patches = plt.hist(ends, bins=25, normed=True)
    plt.xlabel("Final Value of " + security.get_name() + " at end of period.")
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


def predict_percentage_change(name, weeks=52, period="weekly", threshold=0.0, num_bins=10):
    """
    Predicts the percentage change of a security in the next week based on last week's percentage change

    :param name: <String> Ticker name of security (e.g. "AAPL")
    :param weeks: <Int> Weeks since the most recent recorded date (cannot use years/months because months and years
        have varying quantities of days; Numpy requires constancy in datetime arithmetic)
    :param period: <String> Frequency of percentage change data
    :param threshold: <float> Represents the level dividing events A (change >= threshold) & B (change < threshold).
    :param num_bins: <int> Specifies number of bins in the histogram distribution. The more bins, the more realistic
                            the probability distribution will be

    """
    last_row = quandl.get("YAHOO/"+name, rows=1, period=period, order="desc")
    end_date = last_row.index.values[0]  # numpy datetime object
    start_date = end_date - np.timedelta64(weeks, 'W')
    end_date_str = pd.to_datetime(str(end_date))
    end_date_str = end_date_str.strftime('%Y-%m-%d')
    start_date_str = pd.to_datetime(str(start_date))
    start_date_str = start_date_str.strftime('%Y-%m-%d')

    sec = SecurityInfo(name=name, start=start_date_str, end=end_date_str, period=period)
    markov_df = get_data(sec)
    prob_list = percent_change_prob_2x2(markov_df, sec, threshold=threshold)

    pct_change_array = np.array(markov_df["Percentage Change"].dropna())
    pct_above_threshold_array = pct_change_array[pct_change_array >= threshold]
    pct_below_threshold_array = pct_change_array[pct_change_array < threshold]
    n_above, bins_above, patches_above = plt.hist(pct_above_threshold_array, bins=num_bins, normed=True)
    n_below, bins_below, patches_below = plt.hist(pct_below_threshold_array, bins=num_bins, normed=True)

    #print(end_date_str)
    last_percent_change = markov_df.max()[1]
    #print("\nThis week's percentage change was {:.3f}%.".format(last_percent_change))
    # random number used to test whether event A will occur or event B will occur

    next_pct_change_list = []
    for _ in range(10000):
        rand_prob = np.random.random_sample()
        if last_percent_change >= threshold:  # If true, event A occurred
            # prob_list[0][0] is probability that another event A will occur, given event A has happened already
            if rand_prob <= prob_list[0][0]:  # If true, A then A
                next_pct_change = prob_from_bins(n_above, bins_above)
            else:  # If true, A then B
                next_pct_change = prob_from_bins(n_below, bins_below)
        else:  # If true, event B occurred
            # prob_list[1][0] is probability that event A will occur, given event B has happened already
            if rand_prob <= prob_list[1][0]:  # If true, B then A
                next_pct_change = prob_from_bins(n_above, bins_above)
            else:  # If true, B then B
                next_pct_change = prob_from_bins(n_below, bins_below)

        next_pct_change_list.append(next_pct_change)
    next_pct_change_np = np.array(next_pct_change_list)
    plt.clf()
    n, bins, patches = plt.hist(next_pct_change_np, bins=25, normed=True)
    bincenters = 0.5*(bins[1:]+bins[:-1])

    plt.xlabel("Percentage change")
    plt.ylabel("Frequency")

    mean = next_pct_change_np.mean()
    std = next_pct_change_np.std()

    plt.title("Distribution of percentage change in " + name + " Mu: %.3f, Sigma: %.3f"
              % (mean, std), y=1.03)

    # adds vertical lines to the graph corresponding to the x's that represent the number of deviations from the mean
    for num_std_from_mean in range(-3, 4):
        plt.axvline(mean + std * num_std_from_mean)

    # plots the normal pdf of best fit
    y = mlab.normpdf(bincenters, mean, std)
    plt.plot(bincenters, y, 'r--', linewidth=1)

    plt.show()


    return [last_percent_change, mean, std]


def compare_securities_2x2(sec_list, weeks, thresh=0.0):
    """
    Returns an excel sheet with stock name, this week's percentage change, mean of next week's predicted
    percentage change, and standard deviation of next week's predicted percentage change

    :param sec_list: <list> with all the security names
    :param weeks: <int> Number of weeks since the most recent recorded date (cannot use years/months because months and
    years have varying quantities of days; Numpy requires constancy in datetime arithmetic)
    :param thresh: <float> divides percentage changes into two categories (>= and <); applies to each security
    """
    sec_dict = {}
    for name in sec_list:
        sec_info = predict_percentage_change(name, weeks=weeks, threshold=thresh)
        sec_dict[name] = sec_info
    sec_df = pd.DataFrame(sec_dict).transpose()
    sec_df.columns = ['Last % Change', "Mean Predicted % Change", "Standard Deviation " +
                      "Predicted % Change"]
    sec_df= sec_df.sort_values(by=["Mean Predicted % Change"], ascending=True)
    writer = pd.ExcelWriter('output.xlsx')
    sec_df.to_excel(writer, 'Sheet1')
    writer.save()


#compare_securities_2x2(["BAC", "AAPL", "GOOG", "T"], weeks=26, thresh=2.0)

predict_percentage_change("YELP", 104, threshold=5)
#x = SecurityInfo(name="AMZN", start="2009-05-23", end="2016-05-23", period="weekly")
#markov_df = get_data(x)
#print(markov_df)
#percent_change_as_time_plot(markov_df, x)
#percent_change_as_hist(markov_df, x)
#matrix = percent_change_prob_3x3(markov_df, x, lower_thresh=-5, upper_thresh=5)
#rand_walk_3x3_markov(markov_df, matrix, x, lower_thresh=-5, upper_thresh=5)
