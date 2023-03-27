import pandas as pd
from sklearn import metrics
from sklearn.model_selection import KFold, cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def evaluateModel(model, X_test, y_test, isBinary):
  """
    Evaluates the model with 5 metrics and returns expected values and predicted values in a dataframe

    Parameters
    ----------
    model : sklearn model
      model used to make the predictions
    X_test : dataset
      testing features
    y_test : dataset
      testing labels
    isBinary : boolean
      True for binary classes, False for multi classes

    Returns
    -------
    dataframe
      expected values and predicted values in a dataframe
  """
  y_pred = pd.DataFrame(model.predict(X_test))
  print("Accuracy :", metrics.accuracy_score(y_test, y_pred))
  print("Balanced accuracy :", metrics.balanced_accuracy_score(y_test, y_pred))
  print("Confusion matrix :", metrics.confusion_matrix(y_test, y_pred))
  if isBinary:
    print("Precision :", metrics.precision_score(y_test, y_pred))
    print("Recall :", metrics.recall_score(y_test, y_pred))
  else:
    print("Precision :", metrics.precision_score(y_test, y_pred, average='macro', zero_division=0))
    print("Recall :", metrics.recall_score(y_test, y_pred, average='macro', zero_division=0))
  return

def k_fold(classifier, X_data, y_data):
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', classifier())
    ])
    scoring = ['accuracy', 'balanced_accuracy', 'precision_micro', 'recall_micro']
    scores = cross_validate(pipe, X=X_data, y=y_data, cv=kf, scoring=scoring, return_train_score=True)
    print("Accuracy :", scores['test_accuracy'])
    print("Mean Accuracy :", scores['test_accuracy'].mean())
    print("Balanced accuracy :", scores['test_balanced_accuracy'])
    print("Mean Balanced accuracy :", scores['test_balanced_accuracy'].mean())
    print("Precision :", scores['test_precision_micro'])
    print("Mean Precision :", scores['test_precision_micro'].mean())
    print("Recall :", scores['test_recall_micro'])
    print("Mean Recall :", scores['test_recall_micro'].mean())
    return

def stratified_k_fold(classifier, X_data, y_data):
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', classifier())
    ])
    scoring = ['accuracy', 'balanced_accuracy', 'precision_micro', 'recall_micro']
    scores = cross_validate(pipe, X=X_data, y=y_data, cv=kf, scoring=scoring, return_train_score=True)
    print("Accuracy :", scores['test_accuracy'])
    print("Mean Accuracy :", scores['test_accuracy'].mean())
    print("Balanced accuracy :", scores['test_balanced_accuracy'])
    print("Mean Balanced accuracy :", scores['test_balanced_accuracy'].mean())
    print("Precision :", scores['test_precision_micro'])
    print("Mean Precision :", scores['test_precision_micro'].mean())
    print("Recall :", scores['test_recall_micro'])
    print("Mean Recall :", scores['test_recall_micro'].mean())
    return