run import_articlecsv

filtered_ad = articledata;
% removal of some of the features, OpenedDate and StarredDate as they are
% not saturated enough
% Idx and Translations as it is trivial

filtered_ad.OpenedDate = [];
filtered_ad.StarredDate = [];
filtered_ad.Idx = [];
filtered_ad.UserId = [];
%h = histogram(articledata.UserId);
filtered_ad.Domain = [];
%filter out users which only read 5 articles
%filtered_ad(h.Values <5,:) = [];
filtered_ad.ArticleId = [];
filtered_ad.Liked = [];
translations = filtered_ad.Translations;
last_translations = filtered_ad.LastTranslation;
filtered_ad.Translations = [];
filtered_ad.LastTranslation = [];
tree = fitrtree(filtered_ad, last_translations./filtered_ad.Words...
    ,'OptimizeHyperparameters','auto');
view(tree,'Mode','graph')

%users that use translation function: