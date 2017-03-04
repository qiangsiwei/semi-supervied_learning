# -*- coding: utf-8 -*-

import six
import numpy as np
from six.moves import zip
from six.moves import xrange as range
from abc import ABCMeta, abstractmethod
from numpy.core.umath_tests import inner1d
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

import warnings
warnings.filterwarnings("ignore")

###### Base Classifier ######
default_Base_Classifier = DecisionTreeClassifier(max_depth=3) # max_depth=2 max_features=3
# default_Base_Classifier = SVC(kernel='linear',probability=True)
#############################

class BaseEnsemble(object):
    def __init__(self, base_estimator, n_estimators=10,
                 estimator_params=tuple()):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.estimator_params = estimator_params
        self.estimators_ = []

    def _validate_estimator(self, default=None):
        if self.base_estimator is not None:
            self.base_estimator_ = self.base_estimator
        else:
            self.base_estimator_ = default

    def _make_estimator(self, append=True):
        estimator = clone(self.base_estimator_)
        if append:
            self.estimators_.append(estimator)
        return estimator

class BaseWeightBoosting(six.with_metaclass(ABCMeta, BaseEnsemble)):
    @abstractmethod
    def __init__(self,
                 base_estimator=None,
                 n_estimators=50,
                 estimator_params=tuple(),
                 learning_rate=1.,
                 random_state=None):
        super(BaseWeightBoosting, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params)
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None, label_weight=None, cost_matrix=None, cost_matrix2=None):
        if sample_weight is None:
            # Initialize weights to 1 / n_samples
            sample_weight = np.empty(X.shape[0], dtype=np.float)
            sample_weight[:] = 1. / X.shape[0]
        if label_weight is None:
            # Initialize weights to 1 / n_samples
            label_weight = np.empty(X.shape[0], dtype=np.float)
            label_weight[:] = 1.
        # Check parameters
        self._validate_estimator()
        # Clear any previous fit results
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float)
        for iboost in range(self.n_estimators):
            # Boosting step
            if iboost%2 == 0 or cost_matrix2 is None:
                sample_weight, estimator_weight, estimator_error = self._boost(iboost, X, y, sample_weight, label_weight, cost_matrix)
            else:   
                sample_weight, estimator_weight, estimator_error = self._boost(iboost, X, y, sample_weight, label_weight, cost_matrix2)  
            # Early termination
            if sample_weight is None:
                break
            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error
            # Stop if error is zero
            if estimator_error == 0:
                break
            sample_weight_sum = np.sum(sample_weight)
            # Stop if the sum of sample weights has become non-positive
            if sample_weight_sum <= 0:
                break
            if iboost < self.n_estimators - 1:
                # Normalize
                sample_weight /= sample_weight_sum
        return self

    @abstractmethod
    def _boost(self, iboost, X, y, sample_weight):
        pass

class AdaBoostClassifier(BaseWeightBoosting):
    def __init__(self,
                 base_estimator=None,
                 n_estimators=50,
                 learning_rate=1.,
                 random_state=None):
        super(AdaBoostClassifier, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state)

    def fit(self, X, y, sample_weight=None, label_weight=None, cost_matrix=None, cost_matrix2=None):
        return super(AdaBoostClassifier, self).fit(X, y, sample_weight, label_weight, cost_matrix, cost_matrix2)

    def _validate_estimator(self):
        super(AdaBoostClassifier, self)._validate_estimator(
            default=default_Base_Classifier)

    def _boost(self, iboost, X, y, sample_weight, label_weight, cost_matrix):
        return self._boost_discrete(iboost, X, y, sample_weight, label_weight, cost_matrix)
        # return self._boost_real(iboost, X, y, sample_weight)

    def _boost_discrete(self, iboost, X, y, sample_weight, label_weight, cost_matrix):
        estimator = self._make_estimator()
        try:
            estimator.set_params(random_state=self.random_state)
        except ValueError:
            pass
        estimator.fit(X, y, sample_weight=sample_weight)
        y_predict = estimator.predict(X)
        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)
        # Instances incorrectly classified
        incorrect = y_predict != y
        # Instances cost
        if cost_matrix is None:
            sample_cost = [1.]*len(y)
        else:
            sample_cost = [cost_matrix[a][b] for a,b in zip(y,y_predict)]
        # Error fraction
        estimator_error = np.mean(
            np.average(incorrect, weights=sample_weight * sample_cost, axis=0))
        # Stop if classification is perfect
        if estimator_error <= 0:
            return sample_weight, 1., 0.
        n_classes = self.n_classes_
        # Stop if the error is at least as bad as random guessing
        if estimator_error >= 1. - (1. / n_classes):
            self.estimators_.pop(-1)
            if len(self.estimators_) == 0:
                raise ValueError('Ensemble is worse than 1/K')
            return None, None, None
        # Boost weight using multi-class AdaBoost SAMME alg
        estimator_weight = self.learning_rate * (
            np.log((1. - estimator_error) / estimator_error) +
            np.log(n_classes - 1.))
        if not iboost == self.n_estimators - 1:
            # Only boost positive weights
            sample_weight *= sample_cost * label_weight * np.exp(estimator_weight * incorrect *
                                                                ((sample_weight > 0) |
                                                                 (estimator_weight < 0)))
        return sample_weight, estimator_weight, estimator_error

    def predict(self, X):
        pred = self.decision_function(X)
        if self.n_classes_ == 2:
            return self.classes_.take(pred > 0, axis=0)
        return self.classes_.take(np.argmax(pred, axis=1), axis=0)

    def decision_function(self, X):
        n_classes = self.n_classes_
        classes = self.classes_[:, np.newaxis]
        pred = sum((estimator.predict(X) == classes).T * w
                   for estimator, w in zip(self.estimators_,
                                           self.estimator_weights_))
        pred /= self.estimator_weights_.sum()
        if n_classes == 2:
            pred[:, 0] *= -1
            return pred.sum(axis=1)
        return pred
