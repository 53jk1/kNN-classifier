from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from tabulate import tabulate
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest, chi2

from sklearn.decomposition import PCA


def return_results(scores: any):
    mean = np.mean(scores, axis=1)
    std = np.std(scores, axis=1)

    return "%.3f, %.2f" % (mean[clf_id], std[clf_id])


if __name__ == '__main__':

    table = [["", "No filter", "PCA", "chi2"],
             ["GNB", "", "", ""],
             ["kNN", "", "", ""],
             ["CART", "", "", ""]
             ]

    dataset = 'yeast4'
    dataset = np.genfromtxt("%s.csv" % dataset, delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)

    clfs = {
        0: GaussianNB(),
        1: KNeighborsClassifier(),
        2: DecisionTreeClassifier(random_state=21),
    }

    from sklearn.model_selection import StratifiedKFold

    n_splits = 5
    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=21)
    scores = np.zeros((len(clfs), n_splits))

    for fold_id, (train, test) in enumerate(skf.split(X, y)):

        chi_X, chi_y = load_digits(return_X_y=True)
        chi_X_new = SelectKBest(chi2, k=8).fit_transform(X, y)

        for clf_id, clf_name in enumerate(clfs):
            # No filter
            clf = clone(clfs[clf_name])
            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])
            scores[clf_id, fold_id] = accuracy_score(y[test], y_pred)
            table[clf_id+1][1] = return_results(scores)

            # PCA
            pca = PCA()
            pca.fit(X[train])
            pca_X = pca.transform(X)
            clf = clone(clfs[clf_name])
            clf.fit(pca_X[train], y[train])
            y_pred = clf.predict(pca_X[test])
            scores[clf_id, fold_id] = accuracy_score(y[test], y_pred)
            table[clf_id+1][2] = return_results(scores)

            # Chi2
            clf = clone(clfs[clf_name])
            clf.fit(chi_X_new[train], chi_y[train])
            y_pred = clf.predict(chi_X_new[test])
            scores[clf_id, fold_id] = accuracy_score(chi_y[test], y_pred)
            table[clf_id+1][3] = return_results(scores)

    print(tabulate(table))