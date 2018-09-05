import zeeguu
from zeeguu.model import Article, UserActivityData, Bookmark, UserArticleData, UserArticle

session = zeeguu.db.session

visited_url_user_pairs = []


#go over user article instead, unique user/article combinations extracted this way
# translations can be tricky, filtered bookmarks by user then compare bookmarks.text.url with article.url to find intersection and number of translations

for ua in UserArticle.query.all():
    try:
        urlcrop = str(ua.article.url).split('articleURL=')[-1]

        liked = ua.liked
        
        # print(sa.url)
        url_end = urlcrop.find("xtor=RSS")
        if url_end < 0:
            url = str(urlcrop)
        else:
            url = str(urlcrop)[:url_end - 1]

        last_opened = ua.opened

        last_starred = ua.starred

        #apply additional filter over article, url is given in extra data and title!
        translation_events = UserActivityData.find(ua.user, event_filter='UMR - TRANSLATE TEXT',extra_filter = 'title', extra_value = ua.article.title)
        translations = len(translation_events)

        last_word_translated = 0

        # find first occurence of the translated word in the text, this is an indication of the point where the user hooked off
        words = str(ua.article.content).split(" ")
        print(len(words),ua.article.word_count)

        for t in translation_events:
            idx = 0
            for w in words:
                if last_word_translated < idx and w.find(t.value) is not -1:
                    last_word_translated = idx
                idx += 1


        #find latest opened date and closed date
        open_dates = ua.opened
        #closed_dates = UserActivityData.find(bookmark.user, event_filter='UMR - ARTICLE CLOSED')
        """
        time_read = -1
        if closed_dates:
            time_read = 0
            for i in range(len(closed_dates)):
                closed_date = closed_dates[i].time
                j=i
                #find approximate corresponding open_date
                open_date = open_dates[j].time
                while open_date > closed_date:
                    j+=1
                    open_date=open_dates[j+1].time
                print(open_date)
                print(closed_date)
                #time_read += closed_date-open_date
            #print(time_read)
        """


        
        last_starred = ua.starred

        

        ua = UserArticleData.find_or_create(session, ua.user, ua.article,
                                        starred=last_starred,
                                        liked=liked, opened=last_opened,length=ua.article.word_count
                                        ,translated=translations,learned_language=ua.user.learned_language,
                                        domain=ua.article.url.domain,difficulty=ua.article.fk_difficulty,last_word_translated=last_word_translated)
        if last_opened == None:
            print(f'-- Could not find latest opened date {last_starred} x {ua.user.name} x {ua.article.title}')

        #session.commit()
        print(f'SUCCESS: {last_starred} x {ua.user.name} x {ua.article.title}')
    except Exception as ex:
        import traceback

        print(f'-- could not import {urlcrop}')
        print(traceback.format_exc())


