import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics as sm


__all__ = ['show_decision_tree']



def show_decision_tree(df, feature_cols, target_col, test_size=0.2):