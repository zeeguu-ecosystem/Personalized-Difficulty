import zeeguu
from zeeguu.model import UserArticle, User, Language, Article
from zeeguu.language.strategies.word_history_difficulty_estimator import WordHistoryDifficultyEstimator

from plotly import tools
import plotly as py
import plotly.graph_objs as go
from collections import defaultdict
from nltk import SnowballStemmer
from zeeguu.util.text import split_words_from_text
from sqlalchemy.sql.expression import func
import math
import datetime

#userarticles = UserArticle.query.filter(UserArticle.user_id == 534).all()
userarticles = Article.query.order_by(func.rand()).filter(Article.language == Language.find('de')).limit(100).all()
userarticles2 = Article.query.order_by(func.rand()).filter(Article.language == Language.find('fr')).limit(100).all()
print(len(userarticles))
timestamps = defaultdict(list)
score = defaultdict(list)
c = []
i = 0

# create estimators
languages = Language.query.filter(Language.code == 'de').all()
languages = [Language.find('de'), Language.find('fr')]

language_estimator_dict = dict()

#also try recurrence

min_timestamp = 1520729627
max_timestamp = 1528110938

steps = 20

lang_total = dict()
for language in languages:
    articles = UserArticle.query.join(Article).filter(UserArticle.user_id == 534).filter(Article.language == language).all()
    print(language.name, len(articles))
    lang_total[language.name] = len(articles)

data = []



for timestamp in range(min_timestamp, max_timestamp, int((max_timestamp-min_timestamp)/steps)):

    for language in languages:
        #language_estimator_dict[language.name] = \
            #WordHistoryDifficultyEstimator.difficulty_until_timestamp(language, User.find_by_id(534), timestamp)
        language_estimator_dict[language.name] = \
            WordHistoryDifficultyEstimator.difficulty_until_timestamp(language, User.find_by_id(534), timestamp)
    print(timestamp)
    for ua in userarticles:

        if ua == None:
            continue

        result = language_estimator_dict[ua.language.name].estimate_difficulty(ua.content)

        #print(result["normalized"])

        score[ua.language.name].append(result["normalized"])
        timestamps[ua.language.name].append(timestamp)

    for ua in userarticles2:

        if ua == None:
            continue

        result = language_estimator_dict[ua.language.name].estimate_difficulty(ua.content)

        #print(result["normalized"])

        score[ua.language.name].append(result["normalized"])
        timestamps[ua.language.name].append(timestamp)


langs = list(score.keys())
print(langs)

langtitles = [t + " " + str(lang_total[t]) for t in langs]

fig = tools.make_subplots(rows=2, cols = math.ceil(len(langs)/2), subplot_titles=langtitles)

r = 1
c = 1
for lang in langs:
    fig.append_trace(go.Scatter(
        x=[datetime.datetime.fromtimestamp(t) for t in timestamps[lang]],
        y=score[lang],
        mode='markers',
        marker = dict(size= 14,line= dict(width=1), opacity= 0.3))

    ,r,c)
    c+=1
    if c > math.ceil(len(langs)/2):
        c = 1
        r += 1

fig['layout']['yaxis1'].update(range = [0, 1],
                               #title = 'difficulty',
                               title = 'ratio unknown')
fig['layout']['xaxis1'].update(title = 'Date')

for i in range(2,len(langs) + 1):
    print(i)
    fig['layout']['yaxis' + str(i)].update(range = [0, 1])

fig["layout"].update(
        #title= "difficulty over time",
        title= "Difficulty over time",
        hovermode= 'closest',
        showlegend=False,
    width = 1200, height=800
    )


py.offline.plot(fig)