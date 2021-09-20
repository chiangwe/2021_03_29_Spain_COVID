import h2o4gpu;
from sklearn.datasets import load_boston;
import pdb

from h2o4gpu.solvers.linear_regression import LinearRegression
import numpy as np
X, y = load_boston(return_X_y=True);

X = np.vstack( [X for each in range(0, 4000)]).astype(np.float32);
y = np.hstack( [y for each in range(0, 4000)]);


pdb.set_trace()
while(True):
	print("QQQ")
	out = LinearRegression().fit(X, y)
	#out = h2o4gpu.linear_model.LinearRegressionSklearn().fit(X, y)
