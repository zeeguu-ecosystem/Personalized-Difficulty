import csv
import zeeguu


from sqlalchemy import asc
from zeeguu.model.bookmark import Bookmark
from sklearn.manifold import TSNE
from sklearn.linear_model import Ridge
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from imblearn.pipeline import make_pipeline
from matplotlib.colors import ListedColormap
from numpy import mean
from sklearn import preprocessing
from sklearn.decomposition import PCA, KernelPCA
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, SelectFpr, SelectFromModel
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, r2_score, \
    explained_variance_score, mean_squared_error
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score, cross_val_predict, \
    StratifiedKFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
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

langcounter = defaultdict(int)
for obs in observations:
    lang = Article.query.filter(Article.id == int(obs[1])).one().language
    langcounter[lang.name] += 1

print(langcounter)

evaluations = [row[0] for row in observations]
features = [row for row in observations]

# remove user/article id as a feature, fk_diff_db and translation_ratio
for i in reversed(range(len(feature_names))):
    if feature_names[i] in ["user_id", "article_id", "fk_diff_db", "translation_ratio", "evaluation"]:
        feature_names = np.delete(feature_names, i)
        for row in features:
            del row[i]

y_count = Counter(evaluations)
print(Counter(evaluations))

# features
X = np.array(features)
y = np.array(evaluations)

# separate complex from easy features
X_or = np.array(features)
X_simp = np.array([f[:7] for f in features])
X_comp = np.array([f[7:] for f in features])
feature_names_comp = feature_names[7:]
feature_names_simp = feature_names[:7]


name = "Regression"
f1_scores = []
# perform evaluation X times:
features_prev = []
method_scores = dict()

for method in ["simp", "comp", "both", "comb"]:
    for depth in range(1,101):

        f1_scores = []
        clf = Ridge()
        #clf = SVC(kernel='linear')
        sfm = SelectFromModel(clf, threshold=depth/100)
        if method == "simp":
            X_set = X_simp
            feature_set = feature_names_simp

        elif method == "comp":
            X_set = X_comp
            feature_set = feature_names_comp

        elif method == "both":
            X_set = X_or
            feature_set = feature_names

        else:
            X_set = X_comp
            feature_set = feature_names_comp

        X = sfm.fit_transform(StandardScaler().fit_transform(X_set),y)
        cur_features = [feature_set[i] for i in sfm.get_support(indices=True)]

        if method == "comb":
            cur_features = feature_names_simp + cur_features
            X = np.concatenate((X, X_simp), axis=1)

        # skip if features are the same
        if features_prev == cur_features:
            continue
        features_prev = cur_features

        if min(np.shape(X)) == 0:
            break

        # average over different validations
        for k in range(10):

            classifier_pipeline = make_pipeline(StandardScaler(), clf)

            pred = cross_val_predict(classifier_pipeline, X, y, cv=LeaveOneOut())
            f1_scores.append(explained_variance_score(y, pred))

        print(cur_features)
        method_scores[str(cur_features) + " " + str(depth)] = mean(f1_scores, axis=0)

        print(method, depth, mean(f1_scores, axis=0), np.var(f1_scores, axis=0))

sorted_by_value = sorted(method_scores.items(), key=lambda kv: abs(kv[1]), reverse=True)

for method, value in sorted_by_value:
    print(method, value)
