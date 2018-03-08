run import_articlecsv

total = length(articledata.Idx)

total_opened = sum(~isnat(articledata.OpenedDate))
total_starred = sum(~isnat(articledata.StarredDate))

figure(1)
plot(articledata.Words,articledata.Translations,'.')
xlabel('Total words'); ylabel('Total translations');

figure(2)
plot(articledata.Difficulty,articledata.Translations,'.')
xlabel('Difficulty'); ylabel('Total translations');

figure(3)
plot(articledata.Domain,articledata.Translations,'.')
xlabel('Domain'); ylabel('Total Translations')

figure(4)
histogram(articledata.Translations)
xlabel('Translations')

figure(5)
plot(articledata.Words,articledata.LastTranslation./articledata.Words,'.')
xlabel('Total words'); ylabel('Percentage of article read')

figure(6)
plot(articledata.Difficulty,articledata.LastTranslation./articledata.Words,'.')
xlabel('Difficulty'); ylabel('Percentage of article read')
