import pandas as pd

def min_max_normalize(x):
    return (x - x.min()) / (x.max() - x.min())
