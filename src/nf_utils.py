import copy
import sys
from nf_data.compas import compas_data
from nf_data.meps import meps_data
sys.path.append("../")
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import os
import tensorflow as tf
from tensorflow.python.platform import flags
from nf_data.census import census_data
from nf_data.credit import credit_data
from nf_data.bank import bank_data
from utils.utils_tf import model_loss

FLAGS = flags.FLAGS
datasets_dict = {"census": census_data, "credit": credit_data, "bank": bank_data,"compas":compas_data,"meps":meps_data}

def cluster(dataset, cluster_num=4):
    """
    Construct the K-means clustering model to increase the complexity of discrimination
    :param dataset: the name of dataset
    :param cluster_num: the number of clusters to form as well as the number of
            centroids to generate
    :return: the K_means clustering model
    """
    if os.path.exists('../clusters/' + dataset + '.pkl'):
        clf = joblib.load('../clusters/' + dataset + '.pkl')
    else:
        X, Y, input_shape, nb_classes = datasets_dict[dataset]()
        clf = KMeans(n_clusters=cluster_num, random_state=2019).fit(X)
        joblib.dump(clf , '../clusters/' + dataset + '.pkl')
    return clf



def gradient_graph_neuron(x, nx, hidden, nhidden, weights):
    """
    Construct the TF graph of gradient
    :param x: the input placeholder
    :param preds: the model's symbolic output
    :return: the gradient graph
    """
    # Compute loss
    tf_weights = tf.constant(weights, dtype=tf.float32)
    x_loss = model_loss(nhidden*tf_weights, hidden*tf_weights, mean=False)
    nx_loss = model_loss(hidden*tf_weights, nhidden*tf_weights, mean=False)
    x_gradients = tf.gradients(x_loss, x)[0]
    nx_gradients = tf.gradients(nx_loss, nx)[0]
    return x_gradients,nx_gradients


def main(argv=None):
    cluster(dataset=FLAGS.dataset,
            cluster_num=FLAGS.clusters)

if __name__ == '__main__':
    flags.DEFINE_string('dataset', 'census', 'name of datasets')
    flags.DEFINE_integer('clusters', 4, 'number of clusters')
    tf.app.run()
