import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_data():
    """Creates dataframe from log file"""
    # READ DATA #
    dataframe = pd.read_csv('data.csv', skiprows=1)

    # PREPROCESSING DATAFRAME #
    dataframe = dataframe.dropna()  # Delete entries with NaN

    return dataframe


def sort_data(dataframe):
    """
    X: Variables. List with dim = (num_examples, num_parameters). This case (num_examples, 3)
        First item of array: Time since last seen in hours
        Second item of array: Time answering in seconds.
        Third item of array: Time viewing answer in seconds.
    Y: Labels. List with dim = (num_examples, num_classifications). This case (num_examples, 1)
        1: Correct answer.
        0: Incorrect answer.
    """
    # CREATING DICTIONARIES OF VALUES #
    label = dataframe['label'].astype(int).values  # Binary label. Remembered = 1, not remembered = 0
    interval = dataframe['interval'].values  # Time since question last seen in hours
    time_answering = dataframe['answering_time'].values  # Time viewing question before answering in seconds
    time_viewing_answer = dataframe['viewing_answer_time'].values  # Time viewing answer in seconds

    x = list(zip(interval, time_answering, time_viewing_answer))
    y = list(label)
    return x, y


def create_sets(x, y, num_validation, num_test):
    """Creates lists for training and validation. Data is randomly shuffled"""
    x_train, y_train = [], []
    x_val, y_val = [], []
    x_test, y_test = [], []

    num_examples = len(x)
    order = np.random.permutation(num_examples)  # Shuffle data

    for idx in order[0: num_validation]:  # Create validation set
        x_val.append(list(x[idx]))
        y_val.append(y[idx])

    for idx in order[num_validation: num_validation + num_test]:  # Create validation set
        x_test.append(list(x[idx]))
        y_test.append(y[idx])

    for idx in order[num_validation + num_test: num_examples]:  # Create training set
        x_train.append(list(x[idx]))
        y_train.append(y[idx])

    x_train, y_train= np.array(x_train), np.array(y_train)
    x_val, y_val = np.array(x_val), np.array(y_val)
    x_test, y_test = np.array(x_test), np.array(y_test)
    return x_train, y_train, x_val, y_val, x_test, y_test


def plot_data(x, y):
    plot_intervals = [var_list[0] for var_list in x]
    plot_time_viewing_answer = [var_list[2] for var_list in x]
    plot_labels = y

    fig, axes = plt.subplots(nrows=1, ncols=1)

    plt.scatter(x=plot_intervals, y=plot_time_viewing_answer, c=plot_labels, cmap='RdYlGn')

    # TITLE AND LABELS #
    fig.suptitle('Data representation')  # Title
    plt.xlabel("Interval (hours)")
    plt.ylabel("Time viewing answer (seconds)")

    plt.show()
