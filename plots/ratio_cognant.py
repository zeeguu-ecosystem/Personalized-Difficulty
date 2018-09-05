import zeeguu
from zeeguu.model import UserArticle, User, Language, Article
from zeeguu.language.strategies.cognacy_difficulty_estimator import CognacyDifficultyEstimator

from plotly import tools
import plotly as py
import plotly.graph_objs as go
from collections import defaultdict
from nltk import SnowballStemmer
from zeeguu.util.text import split_words_from_text


languages = [Language.find('de')]
#userarticles = UserArticle.query.filter(UserArticle.user_id == 534).all()
userarticles = Article.query.filter(Article.language == languages[0]).all()

print(len(userarticles))
N_words = defaultdict(list)
score = defaultdict(list)
c = []
i = 0

# create estimators


language_estimator_dict = dict()

#also try recurrence
for language in languages:
    language_estimator_dict[language.name] = \
        CognacyDifficultyEstimator.cognacyRatio(language, User.find_by_id(534))


lang_total = dict()
for language in languages:
    articles = Article.query.filter(Article.language == language).all()
    print(language.name, len(articles))
    lang_total[language.name] = len(articles)

data = []

for ua in userarticles:

    if ua == None:
        continue

    result = language_estimator_dict[ua.language.name].estimate_difficulty(ua.content)

    score[ua.language.name].append(result["normalized"])
    N_words[ua.language.name].append(len(ua.content))
    #print(result["normalized"])

langs = list(score.keys())
print(langs)

langtitles = [t + " " + str(lang_total[t]) for t in langs]
import math
fig = tools.make_subplots(rows=1, cols = math.ceil(len(langs)/2), subplot_titles=langtitles)

r = 1
c = 1
for lang in langs:
    fig.append_trace(go.Scatter(
        x=N_words[lang],
        y=score[lang],
        mode='markers',
        marker = dict(size= 14,line= dict(width=1), opacity= 0.3))

    ,r,c)
    c+=1
    if c > math.ceil(len(langs)/2):
        c = 1
        r += 1

fig['layout']['yaxis1'].update(range = [0, 1], title = 'ratio non-cognate')
fig['layout']['xaxis1'].update(title = 'Number of words')

for i in range(2,len(langs) + 1):
    print(i)
    fig['layout']['yaxis' + str(i)].update(range = [0, 1])

fig["layout"].update(
        title= "ratio of non-cognates",
        hovermode= 'closest',showlegend=False,
    width = 1200, height=800,
    )


py.offline.plot(fig)