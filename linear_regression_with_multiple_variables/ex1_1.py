# -*- coding: utf-8 -*-

"""
Author: Santiago Chio
Solution to Exercise 2 of Andrew Ng's Machine Learning course on Coursera
Linear regression with one variable
"""

import numpy as np
import matplotlib.pyplot as plt

DATAFILE = 'ex1data1.txt'
ALPHA = 0.01

def load_data(file_name):
    """
    Read a file and return the data for x and y
    Args:
        file_name: the name of the file.
    Returns:
        a tuple with 2 numpy arrays: x and y.
    """
    x = list()
    y = list()
    with open(file_name, 'r') as f:
        for row in f:
            x_i, y_i = map(float, row.split(','))
            x.append(x_i)
            y.append(y_i)
    return (np.array(x), np.array(y))

def plot_data(x, y):
    """
    Visualice the data in a plot
    """
    plt.plot(x, y, 'bx')
    plt.axis([-5, 24, -5, 25])
    plt.ylabel('Profit in $10,000s')
    plt.xlabel('Population of City in 10,000s')
    plt.savefig('plot_data.png')
    
def plot_data_and_h(x, y, theta):
    """
    Visualice the data in a plot
    """
    plt.plot(x[1,:], y, 'bx', x[1,:], theta.T.dot(x), 'r-')
    plt.axis([-5, 24, -5, 25])
    plt.ylabel('Profit in $10,000s')
    plt.xlabel('Population of City in 10,000s')
    plt.savefig('plot_data_and_h.png')
    
def gradient_descent(x, y, theta):
    """
    Use the batch gradient descent and return the theta vector
    """
    m = y.size
    delta = 1. / m * (np.sum((theta.dot(x) - y) * x, axis=1))
    return theta - ALPHA * delta
    
def cost(x, y, theta):
    m = y.size
    return (1. / (2*m)) * np.sum((theta.dot(x) - y)**2)
    
if __name__ == '__main__':
    ## ------Get and plot the data-------
    x, y = load_data(DATAFILE)
    plot_data(x, y)
    
    ## ------Gradient descent------
    m = y.size # Number of cases
    x = np.array([np.ones(m), x]) # Adding a column of ones
    theta = np.zeros(2) # Initial theta
    past_cost = cost(x, y, theta)
    
    # -----First iteration----
    theta = gradient_descent(x, y, theta)
    plot_data_and_h(x, y, theta)
    new_cost = cost(x, y, theta)
    
    # -----Implementing gradient_descent-----
    while past_cost != new_cost:
        theta = gradient_descent(x, y, theta)
        past_cost, new_cost = new_cost, cost(x, y, theta)
    print('resulting theta: ', theta)
    
    plot_data_and_h(x, y, theta)