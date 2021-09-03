import numpy as np
import sys
sys.path.append("../")
from aif360.datasets.compas_dataset import CompasDataset

def compas_data():
    """
    Prepare the data of dataset Census Income
    :return: X, Y, input shape and number of classes
    """
    cd = CompasDataset()
    # print((cd.feature_names))
    # print(np.shape(cd.features))

    X = np.array(cd.features, dtype=float)
    Y = np.array(cd.labels, dtype=int)

    Y = np.eye(2)[Y.reshape(-1)]
    Y = np.array(Y, dtype=float)

    input_shape = (None, len(X[0]))
    nb_classes = 2
    return X, Y, input_shape, nb_classes


