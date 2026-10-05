import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

class ModelEvaluator():

    @staticmethod
    def evaluate(Y_pred, grid_model, X_test, Y_test):
        best = str(grid_model.best_params_)
        cm = confusion_matrix(Y_test,Y_pred)
        report = classification_report(Y_test,Y_pred)
        roc = roc_auc_score(Y_test,grid_model.predict_proba(X_test)[:,1])
        return best, cm, report, roc

    @staticmethod
    def create_report_table(model_name, Y_test, Y_pred, grid_model, X_test):
        acc = accuracy_score(Y_test, Y_pred)
        prec = precision_score(Y_test, Y_pred, zero_division=0)
        rec = recall_score(Y_test, Y_pred, zero_division=0)
        f1 = f1_score(Y_test, Y_pred, zero_division=0)
        roc = roc_auc_score(Y_test, grid_model.predict_proba(X_test)[:, 1])

        return {
            "Model": model_name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(roc, 4)
        }