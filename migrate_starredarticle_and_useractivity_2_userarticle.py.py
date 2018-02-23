import zeeguu
from zeeguu.model import Article, UserArticle, UserActivityData
from zeeguu.model.starred_article import StarredArticle

session = zeeguu.db.session

for sa in StarredArticle.query.all():
    try:
        article = Article.find_or_create(session, sa.url.as_string())

        likes = UserActivityData.find(sa.user,extra_filter='title', extra_value=str(sa.title),event_filter='UMR - LIKE ARTICLE')
        Nlikes = len(likes)
        #print(sa.url)
        url_end = str(sa.url).find(".html")
        if url_end < 0:
            url = str(sa.url)
        else:
            url = str(sa.url)[:url_end+5]
        last_opened_act = UserActivityData.find_latest(sa.user,extra_filter='articleURL', extra_value=url,event_filter='UMR - OPEN ARTICLE')
        if last_opened_act is None:
            last_opened = None
        else:
            last_opened = last_opened_act.time


        # for debugging, in case latest opened data isn't found
        if last_opened == None and False:
            print()
            print(sa.url)
            activities = UserActivityData.find(sa.user,event_filter='UMR - OPEN ARTICLE')
            print(activities)
            for act in activities:
                print(act.extra_data)
            print()
        
        ua = UserArticle.find_or_create(session, sa.user, article,
                                        starred=sa.starred_date,
                                        liked=Nlikes%2==1, opened=last_opened )
        if last_opened == None:
            print(f'Could not find latest opened date {sa.starred_date} x {ua.user.name} x {ua.article.title}')

        session.commit()
        print(f'{sa.starred_date} x {ua.user.name} x {ua.article.title}')
    except Exception as ex:
        print(f'could not import {sa.url.as_string()}')
        print(ex)
    
            #get user activites from a starredarticle
            #find most recent useractivity that opened the article
            
                