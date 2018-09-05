import zeeguu
from zeeguu.model import UserArticle, User, Language, Article
from zeeguu.language.strategies.word_history_difficulty_estimator import WordHistoryDifficultyEstimator

from plotly import tools
import plotly as py
import plotly.graph_objs as go
from collections import defaultdict
from nltk import SnowballStemmer
from zeeguu.util.text import split_words_from_text

#userarticles = UserArticle.query.filter(UserArticle.user_id == 534).all()
articles = Article.query.all()

print(len(articles))
flesch_kinkaid = defaultdict(list)
score = defaultdict(list)
c = []
i = 0

# create estimators
languages = Language.query.all()

language_estimator_dict = dict()

#also try recurrence
for language in languages:
    language_estimator_dict[language.name] = \
        WordHistoryDifficultyEstimator.difficulty(language, User.find_by_id(534))


lang_total = dict()
for language in languages:
    articles_lang = UserArticle.query.join(Article).filter(UserArticle.user_id == 534).filter(Article.language == language).all()
    print(language.name, len(articles_lang))
    lang_total[language.name] = len(articles_lang)

data = []

for ua in articles:

    if ua == None:
        continue

    stemmer = SnowballStemmer(ua.language.name.lower())

    words = set([stemmer.stem(w.lower()) for w in split_words_from_text(ua.content)])

    result = language_estimator_dict[ua.language.name].estimate_difficulty(ua.content)

    score[ua.language.name].append(result["normalized"])
    flesch_kinkaid[ua.language.name].append(ua.fk_difficulty)
    #print(result["normalized"])

langs = list(score.keys())
print(langs)

langtitles = [t + " " + str(lang_total[t]) for t in langs]
import math
fig = tools.make_subplots(rows=2, cols = math.ceil(len(langs)/2), subplot_titles=langtitles)

r = 1
c = 1
for lang in langs:
    fig.append_trace(go.Scatter(
        x=flesch_kinkaid[lang],
        y=score[lang],
        mode='markers',
        marker = dict(size= 14,line= dict(width=1), opacity= 0.3))

    ,r,c)
    c+=1
    if c > math.ceil(len(langs)/2):
        c = 1
        r += 1

fig['layout']['yaxis1'].update(range = [0, 1], title = 'Difficulty')
fig['layout']['xaxis1'].update(title = 'Flesch-Kinkaid')

for i in range(2,len(langs) + 1):
    print(i)
    fig['layout']['yaxis' + str(i)].update(range = [0, 1])

fig["layout"].update(
        title= "ratio of unknown words",
        hovermode= 'closest'
    )


py.offline.plot(fig)