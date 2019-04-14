import scipy.io
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import sqrt



def mean_squared_error(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true - y_predict)**2) / len(y_true)


def mean_absolute_error(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum(np.absolute(y_true - y_predict)) / len(y_true)


