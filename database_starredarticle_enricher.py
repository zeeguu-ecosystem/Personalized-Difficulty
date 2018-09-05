from datetime import datetime

import zeeguu
from zeeguu import model
from zeeguu.model import User, RSSFeed, Url, Article, DomainName

LOG_CONTEXT = "FEED RETRIEVAL"



def download_from_starrred_article(starArticle: StarredArticle, session):
    """

        Session is needed because this saves stuff to the DB.


    """
    url = str(starArticle.url)
    findart = model.Article.find(url)
    if findart:
        print(f"Already in the DB: {findart}")
    else:
        try:
            
            art = watchmen.article_parser.get_article(url)
            title = art.title
            summary = art.summary
            
            word_count = len(art.text.split(" "))
            
            if word_count < 10:
                zeeguu.log_n_print(f" {LOG_CONTEXT}: Can't find text for: {url}")
            elif word_count < Article.MINIMUM_WORD_COUNT:
                zeeguu.log_n_print(f" {LOG_CONTEXT}: Skipped. Less than {Article.MINIMUM_WORD_COUNT} words of text. {url}")
            else:
                from zeeguu.language.difficulty_estimator_factory import DifficultyEstimatorFactory

                # Create new article and save it to DB
                new_article = model.Article(
                    Url.find_or_create(session, url),
                    title,
                    ', '.join(art.authors),
                    art.text,
                    summary,
                    datetime.now(),
                    RSSFeed.query.first(),
                    starArticle.language
                )
                
                session.add(new_article)
                session.commit()
                zeeguu.log_n_print(f" {LOG_CONTEXT}: Added: {new_article}")
        except:
            import sys
            ex = sys.exc_info()
            zeeguu.log_n_print(f" {LOG_CONTEXT}: Failed to create zeeguu.Article from {url}\n{str(ex)}")
                
if __name__ == "__main__":
    stararts = StarredArticle.query.all()
    
    session = zeeguu.db.session()
    
    
    for i in range(len(stararts)):    
        download_from_starrred_article(stararts[i],session)
    
    session.close()

    #for i in {165..215}; do mysql -u root -e "kill $i" ; done
    #mysqladmin -u root processlist

