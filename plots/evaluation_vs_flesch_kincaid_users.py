import zeeguu
from zeeguu.model import UserArticle, User, Language, Article, UserActivityData, Url, DomainName
from zeeguu.language.strategies.word_history_difficulty_estimator import WordHistoryDifficultyEstimator

from plotly import tools
import plotly as py
import plotly.graph_objs as go
from collections import defaultdict
from nltk import SnowballStemmer
from zeeguu.util.text import split_words_from_text

#userarticles = UserArticle.query.filter(UserArticle.user_id == 534).all()

articles = Article.query.filter().all()
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
		'"finished_difficulty_ok"',
		'"not_finished_for_boring"']

scores = defaultdict(list)
user_list = defaultdict(list)


for eval in diff_evaluations:
    # find top users
    user_freq = defaultdict(int)
    activity = UserActivityData.query.filter(UserActivityData.extra_data == eval).all()
    for act in activity:
        user_freq[act.user] += 1

    sorted_user = sorted(user_freq.items(), key=lambda kv: kv[1], reverse=True)
    users = [u[0] for u in sorted_user[:5]]

    for user in users:
        activity = UserActivityData.query.filter(UserActivityData.extra_data == eval).filter(UserActivityData.user == user).all()

        print(len(activity))
        article_url = []
        for act in activity:
            url = act.value

            for art in articles:
                #print(art.url.domain.domain_name, art.url)
                if art.url.domain.domain_name + art.url.path == url:
                    article_url.append(art)
                    scores[eval].append(art.fk_difficulty)
                    user_list[eval].append(user.name)
                    break

    print(len(article_url), len(activity))

import math
fig = tools.make_subplots(rows=2, cols = math.ceil(len(diff_evaluations)/2), subplot_titles=diff_evaluations)
r = 1
c = 1
for eval in diff_evaluations:
        fig.append_trace(go.Box(y=scores[eval],x = user_list[eval]),r,c)
        c+=1
        if c > math.ceil(len(diff_evaluations)/2):
            c = 1
            r += 1

for i in range(1,len(diff_evaluations) + 1):
    print(i)
    fig['layout']['yaxis' + str(i)].update(range = [0, 100])
    fig['layout']['xaxis' + str(i)].update(title = 'Total evaluations:' + str(len(scores[diff_evaluations[i-1]])))

fig["layout"].update(
        title= "Flesch-kincaid evaluation",
        hovermode= 'closest',
        showlegend=False,
    width = 1200, height=800,
boxmode='group'
    )




py.offline.plot(fig)

#http://www.ilsole24ore.com/art/tecnologie/2018-05-31/chi-sono-e-cosa-fanno-lavoratori-contratto-termine-190823.shtml