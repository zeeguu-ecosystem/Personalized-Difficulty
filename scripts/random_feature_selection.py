import csv
import itertools

import zeeguu
from zeeguu.model.bookmark import Bookmark
from sqlalchemy import asc


from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, ElasticNetCV, LogisticRegression, \
    Perceptron, SGDRegressor, BayesianRidge, LassoCV
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from imblearn.pipeline import make_pipeline
from matplotlib.colors import ListedColormap
from numpy import mean
from sklearn import preprocessing
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, f_regression, SelectFromModel
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, r2_score, mean_squared_error, \
    explained_variance_score
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score, cross_val_predict, \
    StratifiedKFold, KFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVR, NuSVR, SVR
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from zeeguu.model import Article, User, Language, UserActivityData

h = .02  # step size in the mesh

observations = []
time_opened = []

filter_time = False
languageFrom = ""
languageTo = "zh-CN"
user = 534
exclude_user = False
two_features = False

with open('evaluation6.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    firstrow = True
    for row in reader:
        if firstrow:
            firstrow = False
            feature_names = row
        else:

            #filter by language
            if languageFrom != "" and languageTo != "" and (Article.query.filter(Article.id == int(row[1])).one().language != Language.find(languageFrom) \
                or User.query.filter(User.id == int(row[2])).one().native_language != Language.find(languageTo)):
                continue

            # only allow one user
            if not exclude_user and user != 0 and int(row[2]) != user:
                continue

            # filter out one user
            if exclude_user and user != 0 and int(row[2]) == user:
                continue

            art = Article.query.filter(Article.id == int(row[1])).one()

            if filter_time:
                first_time_opened = UserActivityData.query.filter(UserActivityData.user_id == int(row[2])).filter(
                    UserActivityData.value == art.url.domain.domain_name + art.url.path). \
                    filter(UserActivityData.event == "UMR - OPEN ARTICLE").order_by(asc('time')).first().time.timestamp()

                time_opened.append(first_time_opened)

            # filter out a class
            if int(row[-1]) == 3 and two_features:
                continue

            row_numeral = [float(r) for r in row]

            observations.append(row_numeral)

if filter_time:
    time_opened, observations = zip(*sorted(zip(time_opened, observations))[1:])
    print(time_opened)

# remove certain features (what we actually want to predict)
# remove translation ratio

# features to predict
evaluations = [row[0] for row in observations]

# predicting features
features = [row for row in observations]

# remove user/article id as a feature, fk_diff_db and translation_ratio
for i in reversed(range(len(feature_names))):
    if feature_names[i] in ["user_id", "article_id", "fk_diff_db", "translation_ratio", "evaluation"]:
        feature_names = np.delete(feature_names, i)
        for row in features:
            del row[i]

names = ["Ridge"]

classifiers = [Ridge()]

# create datasets
X = np.array(features)
y = np.array(evaluations)

r2_scores = []
for combo in itertools.combinations(range(len(feature_names)), 2):

    print(list(combo))
    X_c = np.array(X[:, list(combo)])
    feature_names_c = [feature_names[c] for c in combo]

    for name, clf in zip(names, classifiers):

        classifier_pipeline = make_pipeline(StandardScaler(),clf)

        predicted = cross_val_predict(classifier_pipeline, X_c, y,cv=KFold(n_splits=10))

        r2_scores.append((feature_names_c, explained_variance_score(y, predicted)))

        print(feature_names_c, explained_variance_score(y, predicted))


print(sorted(r2_scores, key=lambda tup: tup[1]))



