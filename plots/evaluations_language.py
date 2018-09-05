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

for language in Language.query.all():

    articles = Article.query.filter(Article.language == Language.find(language.code)).all()

    scores = defaultdict(int)

    for eval in diff_evaluations:

        activity = UserActivityData.query.join(User).filter(User.native_language == Language.find('en')).filter(UserActivityData.extra_data == eval).all()

        for act in activity:
            url = act.value

            for art in articles:
                #print(art.url.domain.domain_name, art.url)
                if art.url.domain.domain_name + art.url.path == url:
                    scores[eval] += 1
                    break
    print(language.name)
    for eval in diff_evaluations:
        print(eval, scores[eval])
