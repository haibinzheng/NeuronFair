import numpy as np
import sys
sys.path.append("../")
from aif360.datasets.meps_dataset_panel21_fy2016 import MEPSDataset21
cd = MEPSDataset21()
cd.features = np.delete(cd.features, [10], axis=1)

def meps_data():
    """
    Prepare the data of dataset Census Income
    :return: X, Y, input shape and number of classes
    """
    X = np.array(cd.features, dtype=float)
    Y = np.array(cd.labels, dtype=int)
    Y = np.eye(2)[Y.reshape(-1)]
    Y = np.array(Y, dtype=float)
    input_shape = (None, len(X[0]))
    nb_classes = 2
    return X, Y, input_shape, nb_classes

