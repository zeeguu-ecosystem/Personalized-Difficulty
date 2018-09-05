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

print(Counter([int(row[2]) for row in observations]))


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

classifiers = [Ridge()]

# create datasets
X = np.array(features)
X_simp = np.array([f[:7] for f in features])
X_comp = np.array([f[7:] for f in features])
feature_names_comp = feature_names[7:]
y = np.array(evaluations)


figure = plt.figure(figsize=(27, 9))

clf = Ridge()
j = 1

# use only specific features
feature_filter = ['wh_simple_med_s1', 'wh_simple_unique_s20', 'wh_complex_med_s1_c1', 'wh_complex_med_s1_c3', 'wh_complex_med_s3_c3', 'wh_complex_med_unique_s10_c3', 'wh_complex_med_unique_s20_c3', 'wh_complex_med_s1_c5', 'wh_complex_med_unique_s3_c5', 'wh_complex_med_unique_s20_c5', 'wh_complex_med_s1_c10', 'wh_complex_unique_s1_c10', 'wh_complex_med_unique_s3_c10', 'wh_complex_med_unique_s5_c10', 'wh_complex_norm_s1_c20', 'wh_complex_unique_s1_c20', 'wh_complex_med_unique_s5_c20', 'wh_complex_med_unique_s10_c20', 'wh_complex_unique_s20_c20']
for i in reversed(range(len(feature_names))):
    if feature_names[i] not in feature_filter:
        X = np.delete(X, i, 1)
        del feature_names[i]


for name, clf in zip(names, classifiers):

    ax = plt.subplot(1, len(classifiers), j)
    if j == 1:
        ax.set_xlabel('Measured number of translations')
        ax.set_ylabel('Predicted number of translations')
    j += 1

    r2_scores = []
    # perform evaluation X times:
    z = 0


    for i in range(1):

        classifier_pipeline = make_pipeline(StandardScaler(),clf)

        predicted = cross_val_predict(classifier_pipeline, X, y,cv=LeaveOneOut())

        ax.scatter(y, predicted, c="b", alpha=0.2)
        ax.set_xlim(y.min(), y.max())
        ax.set_ylim(y.min(), y.max())
        r2_scores.append(explained_variance_score(y, predicted))
        ax.plot([0, y.max()],[0,y.max()],c='k')
        # print(confusion_matrix(y, pred))
    print(r2_scores)
    print(name, mean(r2_scores), np.var(r2_scores))
plt.savefig('regress.pdf')
plt.show()



