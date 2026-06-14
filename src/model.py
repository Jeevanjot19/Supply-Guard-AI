import xgboost as xgb
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import (roc_auc_score, accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import RandomizedSearchCV

def train_model(X_train, y_train, feature_names, tune: bool = True):
    base = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )

    if tune:
        param_dist = {
            'n_estimators': [200, 300, 400],
            'max_depth': [4, 5, 6, 7],
            'learning_rate': [0.03, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
        }
        search = RandomizedSearchCV(
            base, param_dist, n_iter=20, cv=3,
            scoring='roc_auc', n_jobs=-1, random_state=42, verbose=1
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        print('Best params:', search.best_params_)
    else:
        model = base
        model.fit(X_train, y_train)

    model.feature_names_ = feature_names  # store as plain list attribute
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print('=== Model Evaluation ===')
    print(f'AUC-ROC:  {roc_auc_score(y_test, y_prob):.4f}')
    print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
    print()
    print(classification_report(y_test, y_pred, target_names=['On-Time', 'Late']))
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))

    return y_prob


def save_model(model, path='models/xgb_model.pkl'):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f'Model saved to {path}')


def load_model(path='models/xgb_model.pkl'):
    return joblib.load(path)
