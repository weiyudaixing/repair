import numpy as np
from dask.multiprocessing import get

from torchy.utils import timeit
from sklearn.model_selection import train_test_split
from torchy.learning import learning

if __name__ == '__main__':

    # Create a random dataset
    rng = np.random.RandomState(1)
    X = np.sort(200 * rng.rand(600, 1) - 100, axis=0)
    y = np.array([0 if x > X.mean() or np.random.choice([True, False, False, False]) else 1 for x in X]).T

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        train_size=400,
                                                        random_state=4)

    dsk = {"classifier_type":"RandomForest",
           "RF": (learning.get_classifier, "classifier_type"),
           "X_train" : X_train,
           "X_test"  : X_test,
           "y_train" : y_train,
           "y_test"  : y_test,
           'trained-RF': (learning.train, 'RF', "X_train", "y_train"),
           'predict': (learning.predict, 'trained-RF', 'X_test')}

    with timeit() as time_info:
        output = get(dsk, 'predict')

    print("Time Elapsed: {}".format(time_info.elapsed_time))

