# https://stackoverflow.com/questions/14808945/check-if-variable-is-dataframe
import pandas as pd


def check_variable_type(x):
    if isinstance(x, pd.DataFrame):
        print(f"{type(x)} is a Pandas DataFrame")
