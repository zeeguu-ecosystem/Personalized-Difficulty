import zeeguu
from zeeguu.language.strategies.cognacy_wh_difficulty_estimator import CognacyWordHistoryDifficultyEstimator
from zeeguu.model import UserArticle, User, Language, Article, UserActivityData, Url, DomainName
from zeeguu.language.strategies.word_history_difficulty_estimator import WordHistoryDifficultyEstimator

from plotly import tools
import plotly as py
import plotly.graph_objs as go
from collections import defaultdict, Counter
from nltk import SnowballStemmer
from zeeguu.util.text import split_words_from_text

#userarticles = UserArticle.query.filter(UserArticle.user_id == 534).all()

articles = Article.query.all()

diff_evaluations = ['"finished_difficulty_easy"',
		'"finished_difficulty_hard"',
		'"finished_difficulty_ok"']



scores = defaultdict(list)

activity = UserActivityData.query.filter(UserActivityData.extra_data.in_(diff_evaluations)).limit(10).all()

evals = [act.extra_data for act in activity]

article_url = []
for act in activity:
    url = act.value
    user = act.user
    eval = act.extra_data

    from sqlalchemy import desc, asc
    for art in articles:

        #print(art.url.domain.domain_name, art.url)
        language = art.language
        if art.url.domain.domain_name + art.url.path == url:
            article_url.append(art)
            print(UserArticle.query.filter(UserArticle.user == user).filter(UserArticle.article == art).order_by(
                desc('opened')).all())
            first_time_opened = UserActivityData.query.filter(UserActivityData.user == user).filter(
                UserActivityData.value == url). \
                filter(UserActivityData.event == "UMR - OPEN ARTICLE").order_by(asc('time')).first().time
            #print(int(ua.opened.strftime("%s")))
            diff_est = CognacyWordHistoryDifficultyEstimator.difficulty_until_timestamp(language, user, int(first_time_opened.timestamp()), mode=2, scaling=1)

            difficulty = diff_est.estimate_difficulty(art.content)
            print(difficulty, first_time_opened.timestamp())
            scores[eval].append(difficulty['normalized'])
            break

import math

fig = tools.make_subplots(rows=2, cols = math.ceil(len(diff_evaluations)/2), subplot_titles=diff_evaluations)
r = 1
c = 1
for eval in diff_evaluations:
    fig.append_trace(go.Box(y=scores[eval], name = len(scores[eval])),r,c)

    c+=1
    if c > math.ceil(len(diff_evaluations)/2):
        c = 1
        r += 1

for i in range(1,len(diff_evaluations) + 1):
    print(i)
    fig['layout']['yaxis' + str(i)].update(range = [0, 1])

fig["layout"].update(
        title= "Difficulty evaluation",
        hovermode= 'closest'
    )


py.offline.plot(fig)

#http://www.ilsole24ore.com/art/tecnologie/2018-05-31/chi-sono-e-cosa-fanno-lavoratori-contratto-termine-190823.shtml