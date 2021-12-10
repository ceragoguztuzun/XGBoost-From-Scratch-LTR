import numpy as np
import pandas as pd
import math
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# Research Extension: helps setting the initial prediction as an outcome of a linear model 
def predictInitialPreds(X, y):
    # take a part of the training data
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.9, random_state=440)

    reg = LinearRegression().fit(X_train, y_train)

    init_predictions = []
    init_predictions.append( reg.predict(X_test))
    init_predictions = np.asarray(init_predictions, dtype = float)
    init_predictions = np.concatenate(init_predictions)

    return init_predictions, X_test, y_test

def read_data(path):
    df = pd.read_csv(path, header=None)

    # arranging column names for the dataset.
    df.columns = df.iloc[0]
    df = df.drop(0)
    df = df.reset_index(drop=True)

    return df
    
def enumerate_col(df, col_name):
    col_enum = list(df[col_name])
    d = dict([(y,x+1) for x,y in enumerate(sorted(set(col_enum)))])

    return [d[x] for x in col_enum]

def mean_imputer(df, col_name):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    df[col_name] = imp.fit_transform(df[[col_name]]).ravel()
    
    return df

def gain( hess, grad, lhs, rhs):
    gain = sum(grad[lhs])**2 / (sum(hess[lhs])) + sum(grad[rhs])**2 / (sum(hess[rhs])) - (sum(grad[lhs]) + sum(grad[rhs])**2 / (sum(hess[lhs]) + sum(hess[rhs])))
    
    return gain * 0.5 
    
def getHessian(preds):
    hess = np.empty(preds.shape[0])
    hess.fill(2)

    return hess

def rank_rmse(y: np.ndarray, y_hat: np.ndarray) -> float:
    predictions_sorted, y_test_sorted = zip(*sorted(zip(y_hat, y)))
    
    return mean_squared_error(y_test_sorted, [round(i) for i in predictions_sorted], squared=False)


def rank_accuracy(y: np.ndarray, y_hat: np.ndarray) -> float:

    if y.size != y_hat.size:
        raise ValueError('y and y_hat must be the same shape/size!')

    predictions_sorted, y_test_sorted = zip(*sorted(zip(y_hat, y)))
    
    n = y.size
    return accuracy_score(y_test_sorted, [round(i) for i in predictions_sorted], normalize=False)

def rank_f1_score(y: np.ndarray, y_hat: np.ndarray) -> float:
    if y.size != y_hat.size:
        raise ValueError('y and y_hat must be the same shape/size!')

    predictions_sorted, y_test_sorted = zip(*sorted(zip(y_hat, y)))
    return f1_score(y_test_sorted, [round(i) for i in predictions_sorted], average='macro')


# evaluation metrics
def evaluate(y, y_hat, accuracies = [], rmse_store = []):
    
    accuracy = rank_accuracy(y, y_hat)
    print("Accuracy is: " + str(accuracy))
    accuracies.append(accuracy)

    rmse = rank_rmse(y, y_hat)
    print("RMS error is: " + str(rmse))
    rmse_store.append(rmse)

    f1_score = rank_f1_score(y, y_hat)
    print("F1 score is: " + str(f1_score))

    return accuracies, rmse_store
