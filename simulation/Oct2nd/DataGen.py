### Generate Y following descrete distriution with label 0,1,2 each with constant probability
### p1 = 0.2, p2 = 0.5, p3 = 0.3
### Generate X following normal ditribution
### funture improvement: define pmf
import numpy as np
import pandas as pd

def Simple_generater(number = 1000, label = [0, 1, 2], probss = [0.1,0.3,0.6], probst = [0.5,0.2,0.3]):
    """
    Returns
    -------
    """
    
    # Parameters for Y sample
    labels = label          # possible values of Y
    probss = probss    # probabilities
    number = 1000                    # number of samples

    # Parameters for Y test point
    #labels2 = label      # possible values of Y
    probst = probst     # probabilities 
    
    # Generate Y
    y_sample = np.random.choice(labels, size=number, p=probss)
    y_test = np.random.choice(labels, size=1, p=probst)

    # Parameters for X 
    # Assume sample and test point follow the same distribution)
    ### Y = 0
    m1 = 10       # mean
    sigma1 = 10    # standard deviation
    ### Y = 1
    m2 = 20       # mean
    sigma2 = 10    # standard deviation
    ### Y = 2
    m3 = 30       # mean
    sigma3 = 10    # standard deviation

    # Generate X
    x_sample = np.zeros(len(y_sample))
    for i in range(len(y_sample)):
        if y_sample[i] == 0:
            x_sample[i] = np.random.normal(loc=m1, scale=sigma1, size = 1)
        elif y_sample[i] == 1:
            x_sample[i] = np.random.normal(loc=m2, scale=sigma2, size = 1)
        else:
            x_sample[i] = np.random.normal(loc=m3, scale=sigma3, size = 1)

    x_test = np.zeros(1)
    if y_test == 0:
        x_test = np.random.normal(loc=m1, scale=sigma1, size = 1)
    elif y_test == 1:
        x_test = np.random.normal(loc=m2, scale=sigma2, size = 1)
    else:
        x_test = np.random.normal(loc=m3, scale=sigma3, size = 1)

    # Combine
    # sample = pd.DataFrame({"X": x_sample,"Y": y_sample})
    # test = pd.DataFrame({"X": x_test,"Y": y_test})
    # print(sample[0:10])

    return x_sample, y_sample, x_test, y_test

Simple_generater()

