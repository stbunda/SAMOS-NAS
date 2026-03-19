# implementation based on
# https://github.com/yn-sun/e2epp/blob/master/build_predict_model.py
# and https://github.com/HandingWang/RF-CMOCO
import numpy as np
from numpy.random import RandomState
from sklearn.tree import DecisionTreeRegressor


class CART:
    """ Classification and Regression Tree """
    def __init__(self, n_tree=1000, rnd: RandomState = None):
        self.n_tree = n_tree
        self.name = f'{n_tree} Classification and Regression Trees'
        self.str = f'cart_{self.n_tree}'
        self.model = None
        self.rnd = rnd
        self.metadata = {}

    def __str__(self):
        return self.name

    def _make_decision_trees(self, train_data, train_label, n_tree):
        feature_record = []
        tree_record = []

        for i in range(n_tree):
            sample_idx = np.arange(train_data.shape[0])
            self.rnd.shuffle(sample_idx)
            train_data = train_data[sample_idx, :]
            train_label = train_label[sample_idx]

            feature_idx = np.arange(train_data.shape[1])
            self.rnd.shuffle(feature_idx)
            n_feature = self.rnd.randint(1, train_data.shape[1] + 1)
            selected_feature_ids = feature_idx[0:n_feature]
            feature_record.append(selected_feature_ids)

            dt = DecisionTreeRegressor()
            dt.fit(train_data[:, selected_feature_ids], train_label)
            tree_record.append(dt)

        return tree_record, feature_record

    def fit(self, x, y):
        self.model = self._make_decision_trees(x, y, self.n_tree)

    def predict(self, test_data):
        assert self.model is not None, "carts does not exist, call fit to obtain cart first"

        # redundant variable device
        trees, features = self.model[0], self.model[1]
        test_num, n_tree = len(test_data), len(trees)

        predict_labels = np.zeros((test_num, 1))
        for i in range(test_num):
            this_test_data = test_data[i, :]
            predict_this_list = np.zeros(n_tree)

            for j, (tree, feature) in enumerate(zip(trees, features)):
                predict_this_list[j] = tree.predict([this_test_data[feature]])[0]

            # find the top 100 prediction
            predict_this_list = np.sort(predict_this_list)
            predict_this_list = predict_this_list[::-1]
            this_predict = np.mean(predict_this_list)
            predict_labels[i, 0] = this_predict

        return predict_labels

    def to_config(self):
        self.metadata = {
            "class": self.__class__.__name__,
            "module": self.__class__.__module__,
            "n_tree": self.n_tree,
        }
        return self.metadata