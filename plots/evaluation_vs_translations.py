from datetime import timedelta

import zeeguu
from zeeguu.model import UserArticle, User, Language, Article, UserActivityData, Url, DomainName, Bookmark, Text
from zeeguu.language.strategies.word_history_difficulty_estimator import WordHistoryDifficultyEstimator

from plotly import tools
import plotly as py
import plotly.graph_objs as go
from collections import defaultdict
from nltk import SnowballStemmer
from zeeguu.util.text import split_words_from_text
from sqlalchemy import desc, asc
#userarticles = UserArticle.query.filter(UserArticle.user_id == 534).all()

articles = Article.query.all()

diff_evaluations = ['"finished_difficulty_easy"',
		'"finished_difficulty_hard"',
		'"finished_difficulty_ok"',
		'"not_finished_for_other"',
		'"not_finished_for_boring"',
		'"not_finished_for_broken"',
		'"not_finished_for_difficult"',
		'"maybe_finish_later"']

diff_evaluations = ['"finished_difficulty_easy"',
		'"finished_difficulty_hard"',
		'"finished_difficulty_ok"']



scores = defaultdict(list)

language_estimator_dict = dict()
titles = []

ART_FREQ = defaultdict(int)

#select * from user_activity_data where value like '%/biggest-revelations-from-james-comey-s-new-book-1825321770'
#select * from user_article where article_id = 109309

evaluated_art = []

for eval in diff_evaluations:

    #activity = UserActivityData.query.filter(UserActivityData.extra_data == eval).filter(UserActivityData.user_id == 534).all()
    activity = UserActivityData.query.filter(UserActivityData.extra_data == eval).order_by(asc('time')).all()

    print(len(activity))
    i = 0
    titles.append(eval + " " + str(len(activity)))
    article_url = []
    for act in activity:
        url = act.value
        user = act.user


        for art in articles:

            #print(art.url.domain.domain_name, art.url)
            language = art.language
            if art.url.domain.domain_name + art.url.path == url and str(art.id) + str(act.user) not in evaluated_art:
                i+=1

                bookmarks = Bookmark.query.join(Text).filter(Bookmark.user == user).filter(Text.url == art.url).filter(Bookmark.time <= act.time + timedelta(hours=2)).all()

                evaluated_art.append(str(art.id) + str(act.user))

                words = split_words_from_text(art.content)

                stemmer = SnowballStemmer(language.name.lower())
                words = set([stemmer.stem(w.lower()) for w in words])

                scores[eval].append(len(bookmarks)/len(words))
                break
    print(i)
for k, v in ART_FREQ.items():
    pass
    #print(v, k)

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
    fig['layout']['yaxis' + str(i)].update(range = [0, 0.3])

fig["layout"].update(
        title= "Difficulty evaluation",
        hovermode= 'closest'
    )


py.offline.plot(fig)

#http://www.ilsole24ore.com/art/tecnologie/2018-05-31/chi-sono-e-cosa-fanno-lavoratori-contratto-termine-190823.shtml