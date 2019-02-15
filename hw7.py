"""
Homework 7
ISTA 131

Michelle Monteith
SL: Andrew

This module is a collection of functions that create a March/September DataFrame for the Arctic Sea Ice
Extent. This dataframe has the means and anomalies of both months for each year. The data is then graphed.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def get_Mar_Sept_frame():
    """
    Creates a DataFrame with the means and anomalies for March and September in the years 1979-2017
    :return: DataFrame
    """
    df = pd.read_csv("data_79_17.csv", index_col=0)

    data = []
    for r in df.index:
        mar, sept = [], []
        for i in range(1, 32):
            mar.append(df.loc[r, "03" + str(i).zfill(2)])
        for j in range(1, 31):
            sept.append(df.loc[r, "09" + str(j).zfill(2)])

        data.append([np.mean(mar), 1, np.mean(sept), 1])
    mar_sept = pd.DataFrame(data, index=range(1979, 2018), columns=["March_means", "March_anomalies",
                                                                    "September_means", "September_anomalies"])
    mar_sept["March_anomalies"] = _anomalies(mar_sept["March_means"])
    mar_sept["September_anomalies"] = _anomalies(mar_sept["September_means"])
    return mar_sept


def _anomalies(month):
    """
    Helper function to find the anomalies of each year.
    """
    monthly_mean = month.mean()
    anoms = []
    for i in range(len(month)):
        anoms.append(month.iloc[i] - monthly_mean)
    return anoms


def get_ols_parameters(ser):
    """
    Fits the series to a line by finding the slope, intercept, R^2, and p-value.
    :param ser: Series to fit
    :return: list with the values: [slope, intercept, R^2, p-value]
    """
    slope, intercept, r_value, p_value, std_err = stats.linregress(ser.index, ser.values)
    return [slope, intercept, r_value ** 2, p_value]


def make_prediction(params, description="x-intercept:", x_name="x", y_name="y", ceiling=False):
    """
    Calculates the x-intercept and determines if the result is statistically significant.
    :param params: list of params from get_ols_parameters()
    :param description: string representing what the x-intercept is
    :param x_name: string representing what the x variable is
    :param y_name: string representing what the y variable is
    :param ceiling: should the x-intercept be rounded?
    :return: None
    """

    if ceiling:
        x_int = math.ceil((0 - params[1]) / params[0])
    else:
        x_int = (0 - params[1]) / params[0]

    print(description, str(x_int))
    print(str(int(params[2] * 100)) + "% of variation in " + y_name + " accounted for by " + x_name + " (linear model)")
    print("Significance level of results: " + str(round(params[3] * 100, 1)) + "%")

    if params[3] <= 0.05:
        print("This result is statistically significant.")
    else:
        print("This result is not statistically significant.")
    return None


def make_fig_1(df):
    """
    This function takes a March-September frame and creates a figure that graphs the means for each month
    and the line of best fit for each.
    :param df: Mar Sept DataFrame
    :return: None
    """
    df["March_means"].plot(label="March Mean")
    march_ols = get_ols_parameters(df["March_means"])
    plt.plot(df.index, march_ols[0]*df.index + march_ols[1], color="green")

    df["September_means"].plot(label="September Mean")
    sept_ols = get_ols_parameters(df["September_means"])
    plt.plot(df.index, sept_ols[0]*df.index + sept_ols[1], color="red")

    plt.ylabel("NH Sea Ice Extent ($10^6$ $km^2$)", size=24)
    plt.margins(x=0)
    return None


def make_fig_2(df):
    """
    This function takes a March-September frame and creates a figure that graphs the anomalies for each month
    and the line of best fit for each.
    :param df: Mar Sept DataFrame
    :return: None
    """
    df["March_anomalies"].plot(label="March Anomalies")
    march_ols = get_ols_parameters(df["March_anomalies"])
    plt.plot(df.index, march_ols[0]*df.index + march_ols[1], color="green")

    df["September_anomalies"].plot(label="September Anomalies")
    sept_ols = get_ols_parameters(df["September_anomalies"])
    plt.plot(df.index, sept_ols[0]*df.index + sept_ols[1], color="red")

    plt.ylabel("NH Sea Ice Extent ($10^6$ $km^2$)", size=24)
    plt.title("The Anomaly", size=24)
    plt.margins(x=0)
    return None


def main():
    """
    Get the March-September frame. Get your OLS parameters for the four curves. Make your predictions for winter and
    summer, printing a blank line between them. Make your figures and call plt.show()to draw them. If the predictions
    don't print before the plots show up when main is run, there is a 20-point penalty.
    :return:
    """
    mar_sept = get_Mar_Sept_frame()
    mar_mean_ols = get_ols_parameters(mar_sept["March_means"])
    mar_anom_ols = get_ols_parameters(mar_sept["March_anomalies"])

    sept_mean_ols = get_ols_parameters(mar_sept["September_means"])
    sept_anom_ols = get_ols_parameters(mar_sept["September_anomalies"])

    make_prediction(mar_mean_ols, ceiling=True)
    make_prediction(mar_anom_ols, ceiling=True)

    print()

    make_prediction(sept_mean_ols, ceiling=True)
    make_prediction(sept_anom_ols, ceiling=True)

    make_fig_1(mar_sept)
    plt.figure()
    make_fig_2(mar_sept)
    plt.show()


if __name__ == '__main__':
    main()
