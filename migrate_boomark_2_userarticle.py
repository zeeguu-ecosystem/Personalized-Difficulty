import zeeguu
from zeeguu.model import Article, UserArticle, UserActivityData, Bookmark

session = zeeguu.db.session

visited_url_user_pairs = []

for bookmark in Bookmark.query.all():
    try:

        urlcrop = str(bookmark.text.url).split('articleURL=')[-1]

        url_user_hash = urlcrop + bookmark.user.name

        if url_user_hash in visited_url_user_pairs:
            continue

        visited_url_user_pairs.append(url_user_hash)

        article = Article.find_or_create(session, urlcrop, bookmark.text.language)

        likes = UserActivityData.find(bookmark.user, extra_filter='title', extra_value=str(bookmark.text.url.title), event_filter='UMR - LIKE ARTICLE')
        Nlikes = len(likes)
        #print(sa.url)
        url_end = urlcrop.find("xtor=RSS")
        if url_end < 0:
            url = str(urlcrop)
        else:
            url = str(urlcrop)[:url_end-1]

        last_opened_act = UserActivityData.find(bookmark.user, extra_filter='articleURL', extra_value=url, event_filter='UMR - OPEN ARTICLE', only_latest=True)
        if last_opened_act is None:
            last_opened = None
        else:
            last_opened = last_opened_act.time

        last_starred = None
        last_starred_act = UserActivityData.find(bookmark.user, extra_filter='title', extra_value=bookmark.text.url.title, event_filter='UMR - STAR ARTICLE')
        if len(last_starred_act) %2 == 1:
            last_starred = last_starred_act[0].time

        # for debugging, in case latest opened data isn't found
        if last_opened == None and False:
            print()
            print(urlcrop)
            activities = UserActivityData.find(bookmark.user, event_filter='UMR - OPEN ARTICLE')
            print(activities)
            for act in activities:
                print(act.extra_data)
            print()
        
        ua = UserArticle.find_or_create(session, bookmark.user, article,
                                        starred=last_starred,
                                        liked=Nlikes%2==1, opened=last_opened)
        if last_opened == None:
            print(f'-- Could not find latest opened date {last_starred} x {ua.user.name} x {ua.article.title}')

        session.commit()
        print(f'SUCCESS: {last_starred} x {ua.user.name} x {ua.article.title}')
    except Exception as ex:
        import traceback
        print(f'-- could not import {urlcrop}')
        print(traceback.format_exc())

            
                