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
%filter out users which only read 5 articles
%filtered_ad(h.Values <5,:) = [];
filtered_ad.ArticleId = [];
%filtered_ad.Domain = [];
filtered_ad.Liked = [];
trans_ratio = filtered_ad.LastTranslation./filtered_ad.Words;
filtered_ad.Translations = [];
filtered_ad.LastTranslation = [];
tree = fitctree(filtered_ad, trans_ratio > 0.5,'OptimizeHyperparameters','auto');
view(tree,'Mode','graph')
