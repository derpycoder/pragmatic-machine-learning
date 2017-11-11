"""@author: Abhijit Kar"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

import matplotlib.pyplot as plt
import seaborn as sns

def tp(y_test, y_pred, id):
    """Intersection of the class's row and column"""
    conf_mat = metrics.confusion_matrix(y_test, y_pred)
    return conf_mat[id, id]

def tn(y_test, y_pred, id):
    """Sum of all rows and columns excluding that class's row and column"""
    conf_mat = metrics.confusion_matrix(y_test, y_pred)
    return conf_mat.sum() - conf_mat[id].sum() - conf_mat[:,id].sum() + conf_mat[id, id]

def fp(y_test, y_pred, id):
    """Sum of values in the class's column"""
    conf_mat = metrics.confusion_matrix(y_test, y_pred)
    return conf_mat[:, id].sum() - conf_mat[id, id]

def fn(y_test, y_pred, id):
    """Sum of values in the class's row"""
    conf_mat = metrics.confusion_matrix(y_test, y_pred)
    return conf_mat[id].sum() - conf_mat[id, id]

def recall(y_test, y_pred, id):
    """TP / (TP + FN) - When actual value is positive, How often is the prediction correct?"""
    return tp(y_test, y_pred, id) / float(tp(y_test, y_pred, id) + fn(y_test, y_pred, id))

def specificity(y_test, y_pred, id):
    """TN / (TN + FP) - When the actual value is negative, How often is the prediction correct?"""
    return tn(y_test, y_pred, id) / float(tn(y_test, y_pred, id) + fp(y_test, y_pred, id))

def precision(y_test, y_pred, id):
    """TP / (TP + FP) - When a positive value is predicted, how often is the prediction correct?"""
    return tp(y_test, y_pred, id) / float(tp(y_test, y_pred, id) + fp(y_test, y_pred, id))

func_keys = ['TP', 'TN', 'FN', 'FP', 'Recall', 'Specificity', 'Precision']
func_vals = [tp, tn, fn, fp, recall, specificity, precision]

def show(y_test, y_pred):
    """Returns a Data Frame showcasing the confusion matrix"""
    conf_mat = metrics.confusion_matrix(y_test, y_pred)
    categories = y_test.astype('category').cat.categories
    return pd.DataFrame(conf_mat, columns = categories, index = categories)

def visualize(y_test, y_pred):
    """Draws a HeatMap of the confusion matrix"""
    conf_mat = metrics.confusion_matrix(y_test, y_pred)
    categories = y_test.astype('category').cat.categories
    
    sns.set_context('talk')
    ax = plt.subplot()
    
    sns.heatmap(conf_mat, cmap = 'Blues_r', xticklabels = categories, yticklabels = categories)
    
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted\n')
    plt.ylabel('Actual\n')
    
    ax.title.set_position([0.5, -0.18])
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    
    plt.show()

def describe(y_test, y_pred):
    """Returns a DataFrame with all the metrics calculated from Confusion Matrix at once"""
    categories = y_test.astype('category').cat.categories
    return pd.DataFrame([[func(y_test, y_pred, i) for func in func_vals] for i in range(len(categories))],
                          columns = func_keys, index = categories)

if __name__ == "__main__":
    pima_df = pd.read_csv('../data/pima.csv', dtype = {'diabetes': 'int8'})
    feature_cols = ['num_preg', 'insulin', 'bmi', 'age']
    X = pima_df[feature_cols]
    y = pima_df.diabetes
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)

    print(show(y_test, y_pred))
    visualize(y_test, y_pred)
    print(describe(y_test, y_pred))

    assert (metrics.recall_score(y_test, y_pred) == recall(y_test, y_pred, 1)), "Recall Doesn't Match"
    assert (metrics.precision_score(y_test, y_pred) == precision(y_test, y_pred, 1)), 'Precision Failed'
