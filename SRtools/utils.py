"""
This class provides utility functions for the project.
"""

import numpy as np


def weibull_hazard(t, shape=1, scale=1, translation=0, npoints =100):
    """
    Weibull hazard function. If t is a numpy array, the function will return an array of the same size with the
    corresponding hazard values. if its a list or tupel of lenght 2 it will generate a numpy array with npoints
    between the two values.

    :param t: time
    :param shape: shape parameter
    :param scale: scale parameter
    :param translation: translation parameter
    :param npoints: number of points to generate
    :return: hazard function value
    """
    if isinstance(t, (list, tuple)) and len(t) == 2:
        t = np.linspace(t[0], t[1], npoints)
    return shape/scale * ((t-translation)/scale)**(shape-1) 

def gompertz_makeham_hazard(t, intercept=1, slope=1, makeham=0, npoints=100):
    """
    Gompertz-Makeham hazard function. If t is a numpy array, the function will return an array of the same size with the
    corresponding hazard values. if its a list or tupel of lenght 2 it will generate a numpy array with npoints
    between the two values.

    :param t: time
    :param alpha: alpha parameter
    :param beta: beta parameter
    :param gamma: gamma parameter
    :param translation: translation parameter
    :param npoints: number of points to generate
    :return: hazard function value
    """
    if isinstance(t, (list, tuple)) and len(t) == 2:
        t = np.linspace(t[0], t[1], npoints)
    return intercept*np.exp(slope*(t)) + makeham