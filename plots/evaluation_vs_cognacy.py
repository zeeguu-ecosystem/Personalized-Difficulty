import zeeguu
from zeeguu.language.strategies.cognacy_difficulty_estimator import CognacyDifficultyEstimator
from zeeguu.model import UserArticle, User, Language, Article, UserActivityData, Url, DomainName
from zeeguu.language.strategies.word_history_difficulty_estimator import WordHistoryDifficultyEstimator

from plotly import tools
import plotly as py
import plotly.graph_objs as go
from collections import defaultdict
from nltk import SnowballStemmer
from zeeguu.util.text import split_words_from_text

#userarticles = UserArticle.query.filter(UserArticle.user_id == 534).all()

langFrom = "it"
langTo = "en"

articles = Article.query.filter(Article.language == Language.find(langFrom)).all()
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
language = Language.find('fr')
estimator = CognacyDifficultyEstimator.cognacyRatio(language, User.find_by_id(534))

for eval in diff_evaluations:

    activity = UserActivityData.query.join(User).filter(User.native_language == Language.find(langTo)).filter(UserActivityData.extra_data == eval).all()

    print("activities:",len(activity))
    article_url = []
    for act in activity:
        url = act.value
        print(url)

        for art in articles:
            #print(art.url.domain.domain_name, art.url)
            if art.url.domain.domain_name + art.url.path == url:
                print("article found:", art.id)
                article_url.append(art)
                score = estimator.estimate_difficulty(art.content)
                scores[eval].append(score['unique_ratio'])
                break

print(scores)
titles = ['Easy','Hard','Ok']

import math
fig = tools.make_subplots(rows=1, cols = math.ceil(len(diff_evaluations)), subplot_titles=titles)
r = 1
c = 1
for eval in diff_evaluations:
    fig.append_trace(go.Box(y=scores[eval], name = len(scores[eval])),r,c)

    c+=1
    if c > math.ceil(len(diff_evaluations)/2) and False:
        c = 1
        r += 1



for i in range(1,len(diff_evaluations) + 1):
    print(i)
    fig['layout']['yaxis' + str(i)].update()

fig["layout"].update(
        title= "Ratio-unknown",
        hovermode= 'closest',
        showlegend=False
    )


py.offline.plot(fig)

#http://www.ilsole24ore.com/art/tecnologie/2018-05-31/chi-sono-e-cosa-fanno-lavoratori-contratto-termine-190823.shtml