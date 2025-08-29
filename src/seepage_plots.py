import numpy as np
import pandas as pd


def get_shading_regions(series, threshold):
    """
    Identify and return regions in the series where values are less than a specified threshold.
    """
    regions = []
    start = None
    for i, val in enumerate(series):
        if val < threshold and start is None:
            start = series.index[i]
        elif val >= threshold and start is not None:
            regions.append((start, series.index[i]))
            start = None
    if start is not None:
        regions.append((start, series.index[-1]))
    return regions    



def get_shading_regions2(series1, threshold1, series2, threshold2):
    """
    Identify regions where both input series are below their respective thresholds.

    """
    regions = []
    start = None
    for i, (val1, val2) in enumerate(zip(series1, series2)):
        if val1 < threshold1 and val2 < threshold2 and start is None:
            start = series1.index[i]
        elif (val1 >= threshold1 or val2 >= threshold2) and start is not None:
            regions.append((start, series1.index[i]))
            start = None
    if start is not None:
        regions.append((start, series1.index[-1]))
    return regions

def get_shading_regions_gt(series, threshold):
    """Identify and return regions in the series where values exceed a specified threshold."""
    regions = []
    start = None
    for i, val in enumerate(series):
        if val > threshold and start is None:
            start = series.index[max(i-1, 0)]
        elif val <= threshold and start is not None:
            regions.append((start, series.index[i]))
            start = None
    if start is not None:
        regions.append((start, series.index[-1]))
    return regions      

