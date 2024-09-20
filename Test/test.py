import numpy as np
import pandas as pd
from chemtools.regression import OrdinaryLeastSquares as OLS


x=np.array([0,1,2,3,4,5,6])
y=np.array([0,2,3,5,6,8,10])
ols=OLS()
ols.regression(x,y)
ols.regression_plot()
ols.residual_plot()
ols.summary()