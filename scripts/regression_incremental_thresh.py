import csv
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
        del feature_names[i]
        for row in features:
            del row[i]


names = ["Ridge"]

# filtering features, removing cognates
for i in reversed(range(0, len(feature_names))):
    if "wh" in feature_names[i]:
        for f in features:
            del f[i]
        del feature_names[i]


# features
# = np.array(features)
y = np.array(evaluations)


# separate complex from easy features
X_or = np.array(features)
X_simp = np.array([f[:7] for f in features])
X_comp = np.array([f[7:] for f in features])
feature_names_comp = feature_names[7:]
feature_names_simp = feature_names[:7]
print(y)

figure = plt.figure(figsize=(27, 9))


f1_scores = []

features_prev = []
method_scores = dict()

for method in ["simp", "comp", "both", "comb"]:
    for depth in range(1,101):

        f1_scores = []
        clf = Ridge()
        #clf = SVC(kernel='linear')
        sfm = SelectFromModel(clf, threshold=1/depth)
        if method == "simp":
            sfm.fit(StandardScaler().fit_transform(X_simp), y)
            X = sfm.transform(StandardScaler().fit_transform(X_simp))
            if len(X[0]) == 0:
                continue
            cur_features = [feature_names_simp[i] for i in sfm.get_support(indices=True)]

        elif method == "comp":
            sfm.fit(StandardScaler().fit_transform(X_comp), y)
            X = sfm.transform(StandardScaler().fit_transform(X_comp))
            if len(X[0]) == 0:
                continue
            cur_features = [feature_names_comp[i] for i in sfm.get_support(indices=True)]
        elif method == "both":
            sfm.fit(StandardScaler().fit_transform(X_or), y)
            X = sfm.transform(StandardScaler().fit_transform(X_or))
            if len(X[0]) == 0:
                continue
            cur_features = [feature_names[i] for i in sfm.get_support(indices=True)]
        else:
            sfm.fit(StandardScaler().fit_transform(X_comp), y)
            X = np.concatenate((sfm.transform(StandardScaler().fit_transform(X_comp)), X_simp), axis=1)
            if len(X[0]) == 0:
                continue
            cur_features = [feature_names_comp[i] for i in sfm.get_support(indices=True)]
            cur_features = feature_names_simp + cur_features

        # skip if features are the same
        if features_prev == cur_features:
            continue
        features_prev = cur_features

        if min(np.shape(X)) == 0:
            break


        for k in range(1):

            classifier_pipeline = make_pipeline(StandardScaler(), clf)

            pred = cross_val_predict(classifier_pipeline, X, y, cv=LeaveOneOut())
            f1_scores.append(explained_variance_score(y, pred))

        print(cur_features)
        method_scores[str(cur_features) + " " + str(depth)] = mean(f1_scores, axis=0)

        print(method, depth, mean(f1_scores, axis=0), np.var(f1_scores, axis=0))

sorted_by_value = sorted(method_scores.items(), key=lambda kv: kv[1], reverse=True)

for method, value in sorted_by_value[:200]:
    print(method, value)



