import csv
import zeeguu

from sqlalchemy import asc
from zeeguu.model.bookmark import Bookmark
from sklearn.manifold import TSNE


import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from imblearn.pipeline import make_pipeline
from matplotlib.colors import ListedColormap
from numpy import mean
from sklearn import preprocessing
from sklearn.decomposition import PCA, KernelPCA
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, SelectFpr, SelectFromModel
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score
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


evaluations = [int(row[-1]) for row in observations]
features = [row for row in observations]

# remove user/article id as a feature, fk_diff_db and evaluation
for i in reversed(range(len(feature_names))):
    if feature_names[i] in ["user_id", "article_id", "fk_diff_db", "evaluation", "translation_ratio"]:
        feature_names = np.delete(feature_names, i)
        for row in features:
            del row[i]

y_count = Counter(evaluations)
print(Counter(evaluations))

# features
X = np.array(features)
y = np.array(evaluations)

# separate complex from easy features
X = np.array(features)
X_simp = np.array([f[:7] for f in features])
X_comp = np.array([f[7:] for f in features])
feature_names_comp = feature_names[7:]
feature_names_simp = feature_names[:7]

names = [#"Random Forest",
         "Decision tree","AdaBoost Dec Tree","Linear SVM","RBF SVM"]

classifiers = [
#RandomForestClassifier(max_depth=2),
DecisionTreeClassifier(max_depth=2, class_weight="balanced"),
AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2, class_weight="balanced"),n_estimators=11),
    SVC(kernel="linear", class_weight="balanced"),
    SVC(class_weight="balanced")]

if two_features:

    feature_filter_list = [ ['length', 'av_word_length'],
                            ['length', 'av_word_length', 'wh_cognacy_complex_med_s20_c5'],
                            ['length'],
                            ['med_sentence_length', 'med_word_length', 'fk_diff', 'length', 'unique_length',
                             'av_sentence_length', 'av_word_length', 'wh_simple_norm_s1', 'wh_simple_norm_s3',
                             'wh_cognacy_simple_med_s20'],
                            ['length', 'av_word_length', 'wh_cognacy_complex_med_unique_s20_c20'],
                            ['fk_diff', 'unique_length'],
                            ['fk_diff', 'length', 'unique_length'],
                            ['med_sentence_length', 'med_word_length', 'fk_diff', 'length', 'unique_length',
                             'av_sentence_length', 'av_word_length'],
                            ['av_sentence_length', 'av_word_length'],
                            ['wh_complex_unique_s20_c1', 'wh_complex_norm_s5_c5', 'wh_cognacy_complex_med_s20_c3'],
                            ['wh_complex_unique_s20_c1'],
                            ['av_sentence_length', 'av_word_length', 'wh_complex_unique_s20_c1', 'wh_complex_norm_s3_c5',
                            'wh_complex_norm_s5_c5', 'wh_complex_norm_s3_c10', 'wh_complex_norm_s3_c20'],
                            ['fk_diff', 'length', 'av_sentence_length', 'av_word_length'],
                            ['fk_diff', 'av_sentence_length', 'av_word_length'         ]
                           ]
else:
    feature_filter_list = [['fk_diff', 'length', 'av_word_length'],
                           ['fk_diff', 'length', 'av_word_length', 'wh_complex_unique_s20_c1',
                            'wh_complex_norm_s20_c10', 'wh_cognacy_complex_med_s5_c20',
                            'wh_cognacy_complex_norm_s20_c20'],
                           ['med_sentence_length', 'med_word_length', 'fk_diff', 'length', 'unique_length',
                            'av_sentence_length', 'av_word_length', 'wh_simple_norm_s3',
                            'wh_cognacy_complex_med_unique_s20_c1', 'wh_cognacy_complex_norm_s20_c3'],
                           ['unique_length', 'av_word_length'],
                           ['fk_diff', 'length'],
                           ['med_sentence_length', 'med_word_length', 'fk_diff', 'length', 'unique_length',
                            'av_sentence_length', 'av_word_length', 'wh_simple_unique_s3', 'wh_simple_norm_s10',
                            'wh_complex_norm_s3_c1'],
                           ['med_sentence_length', 'fk_diff', 'length', 'av_word_length', 'wh_complex_med_unique_s1_c1',
                            'wh_complex_norm_s3_c10', 'wh_cognacy_complex_med_s1_c1', 'wh_cognacy_complex_norm_s5_c1',
                            'wh_cognacy_complex_unique_s5_c1', 'wh_cognacy_complex_norm_s3_c3',
                            'wh_cognacy_complex_unique_s1_c5', 'wh_cognacy_complex_norm_s1_c20',
                            'wh_cognacy_complex_med_s10_c20'],
                           ['fk_diff', 'length', 'unique_length', 'av_sentence_length', 'av_word_length'],
                           ['med_sentence_length', 'fk_diff', 'length', 'unique_length', 'av_sentence_length', 'av_word_length'],
                           ['fk_diff', 'unique_length', 'av_sentence_length'],
                           ['wh_simple_unique_s5', 'wh_complex_unique_s10_c1', 'wh_complex_unique_s20_c1',
                            'wh_complex_norm_s3_c20', 'wh_complex_unique_s3_c20', 'wh_cognacy_complex_med_s5_c1',
                            'wh_cognacy_complex_med_unique_s20_c1', 'wh_cognacy_complex_unique_s1_c3',
                            'wh_cognacy_complex_unique_s3_c3'],
                           ['med_word_length', 'length', 'av_sentence_length', 'wh_complex_unique_s20_c1',
                            'wh_complex_norm_s10_c20', 'wh_cognacy_complex_unique_s5_c1',
                            'wh_cognacy_complex_med_unique_s20_c1', 'wh_cognacy_complex_unique_s1_c20'],
                           ['med_sentence_length', 'med_word_length', 'fk_diff', 'length', 'unique_length',
                            'av_sentence_length', 'av_word_length'],
                           ['fk_diff', 'length', 'unique_length', 'av_sentence_length', 'av_word_length']
                           ]

feature_filter_list = ['fk_diff']

for feature_filter in feature_filter_list:
    print()
    print(feature_filter)
    X_copy = X.copy()
    feature_names_copy = feature_names.copy()
    # use only specific features
    for i in reversed(range(len(feature_names))):
        if feature_names_copy[i] not in feature_filter:
            X_copy = np.delete(X_copy, i, 1)
            del feature_names_copy[i]


    for name, clf in zip(names, classifiers):

        f1_scores = []

        # perform evaluation X times:
        for i in range(1):
            #PolynomialFeatures(degree=2)
            #classifier_pipeline = make_pipeline(SelectFpr(f_classif, alpha=0.05),PolynomialFeatures(degree=2),StandardScaler(), PCA(), SMOTE(), clf)

            classifier_pipeline = make_pipeline(StandardScaler(), clf)

            pred = cross_val_predict(classifier_pipeline, X_copy, y, cv=LeaveOneOut())
            f1_scores.append(f1_score(y, pred,average=None))

            #print(confusion_matrix(y, pred))

        print(name, mean(f1_scores,axis=0))