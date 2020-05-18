import pandas as pd
import numpy as np
import plotly.express as px
from plotly.offline import plot
import matplotlib.pyplot as plt

import re as re
from itertools import chain, cycle
from collections import Counter
from scipy import interp

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


def clean_na(data):
    na_pct = df.isnull().sum() / len(df)
    print(na_pct)
    if any(na_pct <= 0.05):
        print("Missing values are BELOW 5%: Extraction conducted")
        for col in data:
            if (na_pct[col] > 0) & (na_pct[col] <= 0.05):
                data.dropna(inplace=True)
    else:
        print("Missing values are ABOVE 5%: Decision should be made manually for each column")


def regex_counter(data, regex):
    data['findall'] = data.text.map(str).apply(lambda x: re.findall(regex, x))
    print('After applying regular expression:', data['findall'])
    hot_words = data.groupby(['language'])['findall'].agg(list)
    hot_words = hot_words.apply(lambda x: list(chain(*x)))
    hot_words = hot_words.apply(lambda x: Counter(x))
    d = dict(hot_words)
    #print("Hot Words Counter: ", d)
    return d


def top_n_words_feature_generator(data, dict_counter, top_n):
    hw = pd.DataFrame([(k, i, j) for k, v in dict_counter.items() for i, j in v.items()],
                      columns=['language', 'hot_word', 'freq'])
    print(hw.head())

    hw = hw.sort_values(['language', 'freq'], ascending=False)
    hot_words_list = hw.groupby('language').head(top_n)['hot_word']

    #print("Hot_words_list per language", hot_words_list)

    def check(rex):
        if rex:
            return 1
        else:
            return 0

    for h in hot_words_list.unique():
        name_var = h + '_flag'
        data[name_var] = data['text'].map(str).apply(lambda x: check(re.findall(h, x)))
    print("Flagged dataset", data.head())
    return data


def multi_roc_plot(clf, x_train, x_test, y_train, y_test):

    n_classes = y.nunique()
    y_labels = np.unique(y)

    y_score = clf.fit(x_train, y_train).decision_function(x_test)

    y_train = label_binarize(y_train, classes=y_labels)
    y_test = label_binarize(y_test, classes=y_labels)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[y_labels[i]] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Interpolation of ROC curves for smoothness
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    lw = 2
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'cornflowerblue','darkorange'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(y_labels[i], roc_auc[y_labels[i]]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC multi-class')
    plt.legend(loc="lower right")
    plt.show()


def optimal_tresholds(clf, x_train, x_test, y_train, y_test):
    n_classes = y.nunique()
    y_labels = np.unique(y)

    y_score = clf.fit(x_train, y_train).decision_function(x_test)

    y_train = label_binarize(y_train, classes=y_labels)
    y_test = label_binarize(y_test, classes=y_labels)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    trsh = dict()
    opti_trsh = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], trsh[i] = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[y_labels[i]] = auc(fpr[i], tpr[i])
        print(tpr[i] - fpr[i])
        print(trsh[i])
        opti_idx = np.argmax(tpr[i] - fpr[i])
        opti_trsh[i] = trsh[i][opti_idx]
    return opti_trsh


if __name__ == '__main__':
    df = pd.read_csv("lang_data.csv")

    print(df.shape)
    clean_na(df)
    print(df.shape)

    eda1 = df.groupby("language").count()
    print(eda1)

    d = regex_counter(df, r'[a-zA-Z]{2,}')

    df = top_n_words_feature_generator(df, d, 25)

    # Model -------------------------------
    df = df.drop(['text', 'findall'], axis=1)
    y = df.language
    print('Y_SHAPE', y.nunique())
    x = df.drop(['language'], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(df.drop(['language'], axis=1), df.language, random_state=48,
                                                        stratify=df.language, test_size=0.3)

    lr = LogisticRegression(multi_class='auto', solver='lbfgs')
    lr.fit(x_train, y_train)

    # Summary ------------------------------
    print('TRAIN:',lr.score(x_train, y_train), 'TEST:', lr.score(x_test, y_test))
    print('Conf_Matrix TRAIN: \n', confusion_matrix(y_train, lr.predict(x_train)))
    print('Conf_Matrix TEST: \n', confusion_matrix(y_test, lr.predict(x_test)))

    multi_roc_plot(lr,x_train, x_test, y_train, y_test)

