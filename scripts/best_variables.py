import string
from datetime import timedelta

import re
import sqlalchemy

import zeeguu
from collections import Counter
from sqlalchemy import asc
from zeeguu.model import UserActivityData

from zeeguu.model.bookmark import Bookmark
from numpy import delete, array_split, dtype
import csv

import matplotlib.pyplot as plt
from numpy.ma import array, log10

from sklearn.decomposition import PCA
from sklearn.feature_selection import  f_classif, SelectKBest

from zeeguu.model.article import Article
from zeeguu.model.language import Language
from zeeguu.model.user import User

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


print(Counter([int(o[2]) for o in observations]))
print(Counter([Article.query.filter(Article.id == int(o[1])).one().language.name
               +User.query.filter(User.id == int(o[2])).one().native_language.name for o in observations]))

if filter_time:
    time_opened, observations = zip(*sorted(zip(time_opened, observations))[25:])
    print(time_opened)

# split in evaluations and features
evaluations = [row[-1] for row in observations]
features = [row for row in observations]
feature_names = feature_names


# remove user/article id as a feature, fk_diff_db and evaluation
for i in reversed(range(len(feature_names))):
    if feature_names[i] in ["user_id", "article_id", "fk_diff_db", "evaluation"]:
        feature_names = delete(feature_names, i)
        for row in features:
            del row[i]

X = array(features)
y = array(evaluations)


# #############################################################################
# calculate score for each feature
selector = SelectKBest(f_classif, k=1)
selector.fit(X,y)
scores = selector.pvalues_

# filter out features with confidence lower than 95%
for i in reversed(range(len(scores))):
    if scores[i] > 0.05:
        scores = delete(scores, i)
        feature_names = delete(feature_names, i)

# convert to log scale
score_log = []
for score in scores:
    score_log.append(-log10(score))

# sort in descending order
score_log, feature_names = zip(*sorted(zip(score_log, feature_names),reverse=True))

# take top X
score_log = score_log[:50]
feature_names = feature_names[:50]

# sort ascending order
score_log, feature_names = zip(*sorted(zip(score_log, feature_names)))

plots = 4
split_sets = array_split(score_log, plots)
name_split = array_split(feature_names, plots)

threshold = -log10(0.05)
for i in range(plots):

    names = name_split[i]
    print(len(names))
    score_logs = split_sets[i]

    ylist = [threshold for f in names]

    plt.figure(i)
    plt.clf()
    plt.plot(ylist, names, "k--")

    plt.barh(names, score_logs,
            label=r'$-log(p_{value})$', color='blue',
            edgecolor='black')


    #plt.title("Comparing feature selection")
    plt.xlabel(r'$-log(p_{value})$')
    plt.ylabel('Feature')
    plt.xticks()
    #plt.yticks(())
    plt.axis('tight')
    plt.subplots_adjust(left = 0.6)


plt.show()

