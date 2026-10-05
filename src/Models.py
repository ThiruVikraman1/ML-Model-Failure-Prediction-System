import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier


class Classification_Models:

    @staticmethod
    def _extract_probabilities(grid, X_test):
        best_est = grid.best_estimator_
        if hasattr(best_est, "predict_proba"):
            return best_est.predict_proba(X_test)
        elif hasattr(best_est, "decision_function"):
            decision = best_est.decision_function(X_test)
            return 1.0 / (1.0 + np.exp(-decision))
        return None

    @staticmethod
    def logistic(X_train, Y_train, X_test, scoring='f1_weighted'):       
        param_grid = {
            'solver': ['lbfgs', 'newton-cg', 'liblinear', 'saga'],
            'penalty': ['l2'],
            'C': [0.01, 0.1, 1, 10],
            'class_weight': [None, 'balanced']
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), param_grid, refit=True, verbose=1, scoring=scoring, cv=cv, n_jobs=-1)
        grid.fit(X_train, Y_train)
        y_pred = grid.predict(X_test)
        y_prob = Classification_Models._extract_probabilities(grid, X_test)
        return y_pred, y_prob, grid

    @staticmethod
    def SVM(X_train, Y_train, X_test, scoring='f1_weighted'):
        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf', 'linear'],
            'class_weight': [None, 'balanced']
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid = GridSearchCV(SVC(probability=True, random_state=42), param_grid, refit=True, verbose=1, scoring=scoring, cv=cv, n_jobs=-1)
        grid.fit(X_train, Y_train)
        y_pred = grid.predict(X_test)
        y_prob = Classification_Models._extract_probabilities(grid, X_test)
        return y_pred, y_prob, grid

    @staticmethod
    def KNN(X_train, Y_train, X_test, scoring='f1_weighted'):
        param_grid = {
            'n_neighbors': [3, 5, 7, 11],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'kd_tree', 'ball_tree', 'brute']
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid = GridSearchCV(KNeighborsClassifier(), param_grid, refit=True, verbose=1, scoring=scoring, cv=cv, n_jobs=-1)
        grid.fit(X_train, Y_train)
        y_pred = grid.predict(X_test)
        y_prob = Classification_Models._extract_probabilities(grid, X_test)
        return y_pred, y_prob, grid

    @staticmethod
    def NaiveBayes(X_train, Y_train, X_test, scoring='f1_weighted'):
        param_grid = {
            'var_smoothing': np.logspace(0, -9, num=20)
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid = GridSearchCV(GaussianNB(), param_grid, refit=True, verbose=1, scoring=scoring, cv=cv, n_jobs=-1)
        grid.fit(X_train, Y_train)
        y_pred = grid.predict(X_test)
        y_prob = Classification_Models._extract_probabilities(grid, X_test)
        return y_pred, y_prob, grid

    @staticmethod
    def DecisionTree(X_train, Y_train, X_test, scoring='f1_weighted'):
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_depth': [3, 5, 10, None],
            'max_features': ['sqrt', 'log2', None],
            'class_weight': [None, 'balanced']
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, refit=True, verbose=1, scoring=scoring, cv=cv, n_jobs=-1)
        grid.fit(X_train, Y_train)
        y_pred = grid.predict(X_test)
        y_prob = Classification_Models._extract_probabilities(grid, X_test)
        return y_pred, y_prob, grid

    @staticmethod
    def RandomForest(X_train, Y_train, X_test, scoring='f1_weighted'):
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, None],
            'class_weight': ['balanced', 'balanced_subsample']
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, refit=True, verbose=1, scoring=scoring, cv=cv, n_jobs=-1)
        grid.fit(X_train, Y_train)   
        y_pred = grid.predict(X_test)
        y_prob = Classification_Models._extract_probabilities(grid, X_test)
        return y_pred, y_prob, grid

    @staticmethod
    def ExtraTrees(X_train, Y_train, X_test, scoring='f1_weighted'):
        param_grid = {
            'n_estimators': [50, 100, 200],
            'criterion': ['gini', 'entropy'],
            'max_depth': [5, 10, None],
            'class_weight': [None, 'balanced']
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid = GridSearchCV(ExtraTreesClassifier(random_state=42), param_grid, refit=True, verbose=1, scoring=scoring, cv=cv, n_jobs=-1)
        grid.fit(X_train, Y_train)
        y_pred = grid.predict(X_test)
        y_prob = Classification_Models._extract_probabilities(grid, X_test)
        return y_pred, y_prob, grid

    @staticmethod
    def AdaBoost(X_train, Y_train, X_test, scoring='f1_weighted'):
        param_grid = {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.05, 0.1, 1.0]
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid = GridSearchCV(AdaBoostClassifier(random_state=42), param_grid, refit=True, verbose=1, scoring=scoring, cv=cv, n_jobs=-1)
        grid.fit(X_train, Y_train)
        y_pred = grid.predict(X_test)
        y_prob = Classification_Models._extract_probabilities(grid, X_test)
        return y_pred, y_prob, grid

    @staticmethod
    def GradientBoosting(X_train, Y_train, X_test, scoring='f1_weighted'):
        param_grid = {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5]
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid, refit=True, verbose=1, scoring=scoring, cv=cv, n_jobs=-1)
        grid.fit(X_train, Y_train)
        y_pred = grid.predict(X_test)
        y_prob = Classification_Models._extract_probabilities(grid, X_test)
        return y_pred, y_prob, grid

    @staticmethod
    def LightGBM(X_train, Y_train, X_test, scoring='f1_weighted', pos_scale=1.0):
        param_grid = {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [15, 31],
            'max_depth': [3, 5, -1],
            'scale_pos_weight': [1.0, pos_scale]
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid = GridSearchCV(lgb.LGBMClassifier(random_state=42, verbose=-1), param_grid, refit=True, verbose=1, scoring=scoring, cv=cv, n_jobs=-1)
        grid.fit(X_train, Y_train)
        y_pred = grid.predict(X_test)
        y_prob = Classification_Models._extract_probabilities(grid, X_test)
        return y_pred, y_prob, grid

    @staticmethod
    def XGBoost(X_train, Y_train, X_test, scoring='f1_weighted', pos_scale=1.0):
        param_grid = {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5],
            'scale_pos_weight': [1.0, pos_scale]
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid = GridSearchCV(xgb.XGBClassifier(eval_metric='logloss', random_state=42), param_grid, refit=True, verbose=1, scoring=scoring, cv=cv, n_jobs=-1)
        grid.fit(X_train, Y_train)
        y_pred = grid.predict(X_test)
        y_prob = Classification_Models._extract_probabilities(grid, X_test)
        return y_pred, y_prob, grid

    @staticmethod
    def CatBoost(X_train, Y_train, X_test, scoring='f1_weighted'):
        param_grid = {
            'iterations': [50, 100, 150],
            'learning_rate': [0.01, 0.05, 0.1],
            'depth': [4, 6]
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid = GridSearchCV(CatBoostClassifier(verbose=0, random_state=42), param_grid, refit=True, verbose=1, scoring=scoring, cv=cv, n_jobs=-1)
        grid.fit(X_train, Y_train)
        y_pred = grid.predict(X_test)
        y_prob = Classification_Models._extract_probabilities(grid, X_test)
        return y_pred, y_prob, grid