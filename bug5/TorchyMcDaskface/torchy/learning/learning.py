"""This file contains all classifier training and prediction functions"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from torchy import utils
import vigra
import h5py
import time
import cPickle as pickle

def get_classifier(classifier_type, filename=None):
    if classifier_type == 'RandomForest':
        if filename is not None:
            print("loading sklearn forest from %s"%filename)
            return pickle.load(open(filename,"rb"))
        rf = RandomForestClassifier(max_depth=30)
        print("Warning: random forest is trainged with random data...only for debug")
        rf.fit(np.random.rand(20,17), np.random.choice([0,1,2,3],size=20))
        return rf
    elif classifier_type == 'VigraRF':
        if filename is not None:
            return vigra.learning.RandomForest(filename, 'Forest0000')
        return vigra.learning.RandomForest()
    else:
        raise NotImplementedError()

def train(classifier, X_train, y_train):
    classifier.fit(X_train, y_train)
    return classifier

def load_classifier(classifier):
    return classifier

def count_labels(classifier):
    if isinstance(classifier, vigra.learning.RandomForest):
        return classifier.labelCount()
    elif isinstance(classifier, RandomForestClassifier):
        return classifier.n_classes_
    else:
        raise NotImplementedError()

def image_prediction(classifier_type, classifier_file_name, features, request):
    s_list, f_list = utils.get_feature_index(request)
    classifier = get_classifier(classifier_type, classifier_file_name)
    export_shape = list(features.shape)
    export_shape[0] = 1
    export_shape[1] = count_labels(classifier)
    predshape = [len(s_list), np.product(features.shape[2:])]
    f = features[s_list, f_list].reshape(predshape).astype(np.float32)
    # write to file
    write_output(predict(classifier, f.T).reshape(export_shape), request)
    return "done"

def predict(classifier, X_test):
    if isinstance(classifier, vigra.learning.RandomForest):
        return classifier.predictProbabilities(X_test)
    elif isinstance(classifier, RandomForestClassifier):
        return classifier.predict_proba(X_test)
    else:
        raise NotImplementedError()


def write_output(data, request):
    print("try writing",request["output_file_name"],request["roi_with_halo"])
    with h5py.File(request["output_file_name"], "r+") as f:
        print(f["data"].shape)
        print(data.shape)
        print(f["data"][request["roi_output"]].shape)
        hs = request["halo_size"]
        f["data"][request["roi_output"]] = data
    return "done"