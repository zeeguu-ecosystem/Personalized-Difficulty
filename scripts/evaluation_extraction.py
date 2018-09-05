import zeeguu
from zeeguu.model.bookmark import Bookmark

from datetime import timedelta

from zeeguu.language.strategies.cognacy_wh_difficulty_estimator import CognacyWordHistoryDifficultyEstimator
from zeeguu.model.user import User


from zeeguu.model.text import Text
from zeeguu.model.language import Language
from zeeguu.model.user_activitiy_data import UserActivityData
from zeeguu.model.article import Article
from zeeguu.language.strategies.cognacy_difficulty_estimator import CognacyDifficultyEstimator
from zeeguu.language.strategies.flesch_kincaid_difficulty_estimator import FleschKincaidDifficultyEstimator

from zeeguu.language.strategies.word_history_difficulty_estimator import WordHistoryDifficultyEstimator

from collections import defaultdict
from zeeguu.util import text
from sqlalchemy import desc, asc
import csv

"""
    script used for extracting evaluations with corresponding features from the current database
    used for analysis of the database

"""

fileName = 'evaluation7.csv'

articles = Article.query.all()

diff_evaluations = ['"finished_difficulty_easy"',
		'"finished_difficulty_hard"',
		'"finished_difficulty_ok"']

observations = defaultdict(list)
evaluated_art = set()

# map evaluation to number
eval_dict = {'"finished_difficulty_easy"': 1,
             '"finished_difficulty_hard"': 3,
             '"finished_difficulty_ok"': 2}

activity = UserActivityData.query.filter(UserActivityData.extra_data.in_(diff_evaluations)).order_by(asc('time')).all()

print(len(activity))
i = 0

for act in activity:
    url = act.value
    user = act.user
    print(i)
    i+=1

    for art in articles:

        language = art.language
        if art.url.domain.domain_name + art.url.path == url and str(art.id) + str(act.user) not in evaluated_art:

            bookmarks = Bookmark.query.join(Text).filter(Bookmark.user == user).filter(Text.url == art.url).filter(
                Bookmark.time <= act.time + timedelta(hours=2)).all()

            evaluated_art.add(str(art.id) + str(act.user))

            words = text.split_unique_words_from_text(art.content, language)

            observations["translation_ratio"].append(len(bookmarks)/len(words))
            observations["article_id"].append(art.id)
            observations["user_id"].append(user.id)
            # fk_diff_db is the Flesch-Kincaid score as stored in the db which was calculated differently
            # compared to now
            observations["fk_diff_db"].append(art.fk_difficulty)
            observations["fk_diff"].append(
                FleschKincaidDifficultyEstimator.estimate_difficulty(art.content, language, None)["grade"])
            observations["length"].append(text.length(art.content))
            observations["unique_length"].append(text.unique_length(art.content,language))
            observations["av_sentence_length"].append(text.average_sentence_length(art.content))
            observations["av_word_length"].append(text.average_word_length(art.content, language))
            observations["med_word_length"].append(text.median_word_length(art.content, language))
            observations["med_sentence_length"].append(text.median_sentence_length(art.content))

            # WordHistoryDifficultyEstimator, based on seen events
            first_time_opened = UserActivityData.query.filter(UserActivityData.user == user).filter(
                UserActivityData.value == url). \
                filter(UserActivityData.event == "UMR - OPEN ARTICLE").order_by(asc('time')).first().time

            for s in [1, 3, 5, 10, 20]:
                estimator = WordHistoryDifficultyEstimator.difficulty_until_timestamp(art.language, user,
                                                                                      int(
                                                                                          first_time_opened.timestamp()),
                                                                                      mode=2, scaling=s)
                difficulty = estimator.estimate_difficulty(art.content)
                observations["wh_simple_norm_s"+str(s)].append(difficulty["normalized"])
                observations["wh_simple_med_s" + str(s)].append(difficulty["median"])
                observations["wh_simple_med_unique_s" + str(s)].append(difficulty["median_unique"])
                observations["wh_simple_unique_s" + str(s)].append(difficulty["unique_ratio"])

            # WordHistoryDifficultyEstimator, based on seen and context events
            for s in [1,3,5,10,20]:
                for c in [1,3,5,10,20]:
                    estimator = WordHistoryDifficultyEstimator.difficulty_until_timestamp(art.language, user,
                                                                                          int(
                                                                                              first_time_opened.timestamp()),
                                                                                          mode=1, scaling=c, scaling2=s)
                    difficulty = estimator.estimate_difficulty(art.content)
                    observations["wh_complex_norm_s" + str(s) + "_c"+str(c)].append(difficulty["normalized"])
                    observations["wh_complex_med_s" + str(s) + "_c"+str(c)].append(difficulty["median"])
                    observations["wh_complex_med_unique_s" + str(s) + "_c"+str(c)].append(difficulty["median_unique"])
                    observations["wh_complex_unique_s" + str(s) + "_c"+str(c)].append(difficulty["unique_ratio"])

            # CognacyWordHistoryDifficultyEstimator, based on context events
            for s in [1,3,5,10,20]:
                estimator = CognacyWordHistoryDifficultyEstimator.difficulty_until_timestamp(art.language, user,
                                                                                      int(first_time_opened.timestamp()),
                                                                                      mode=2, scaling=s)
                difficulty = estimator.estimate_difficulty(art.content)
                observations["wh_cognacy_simple_norm_s" + str(s)].append(difficulty["normalized"])
                observations["wh_cognacy_simple_med_s" + str(s)].append(difficulty["median"])
                observations["wh_cognacy_simple_med_unique_s" + str(s)].append(difficulty["median_unique"])
                observations["wh_cognacy_simple_unique_s" + str(s)].append(difficulty["unique_ratio"])

            # CognacyWordHistoryDifficultyEstimator, based on seen and context events
            for s in [1,3,5,10,20]:
                for c in [1,3,5,10,20]:
                    estimator = CognacyWordHistoryDifficultyEstimator.difficulty_until_timestamp(art.language, user,
                                                                                          int(
                                                                                              first_time_opened.timestamp()),
                                                                                          mode=1, scaling=c, scaling2=s)
                    difficulty = estimator.estimate_difficulty(art.content)
                    observations["wh_cognacy_complex_norm_s" + str(s) + "_c"+str(c)].append(difficulty["normalized"])
                    observations["wh_cognacy_complex_med_s" + str(s) + "_c"+str(c)].append(difficulty["median"])
                    observations["wh_cognacy_complex_med_unique_s" + str(s) + "_c"+str(c)].append(difficulty["median_unique"])
                    observations["wh_cognacy_complex_unique_s" + str(s) + "_c"+str(c)].append(difficulty["unique_ratio"])

            # CognacyDifficultyEstimator
            estimator = CognacyDifficultyEstimator(language,user)
            estimation = estimator.estimate_difficulty(art.content)

            observations["cognacy_norm"].append(difficulty["normalized"])
            observations["cognacy_med"].append(difficulty["median"])
            observations["cognacy_med_unique"].append(difficulty["median_unique"])
            observations["cognacy_unique"].append(difficulty["unique_ratio"])

            observations["evaluation"].append(eval_dict[act.extra_data])
            break

# transform dict to a list with each row representing an evaluation and each column a feature
observation_list = []
for i in range(len(list(observations.values())[0])):
    observation_list.append([feat[i] for feat in observations.values()])

with open(fileName, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(list(observations.keys()))
    for observation in observation_list:
        writer.writerow(observation)