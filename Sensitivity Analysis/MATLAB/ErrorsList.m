% Function to see which patients show errors in ROIs

% load scores!
indx = 1;
thesescores.PatNums = scores(indx).PatNums;
thesescores.biopsyresults = (scores(indx).biopsyresults);
thesescores.scores = transpose(scores(indx).scores);

% PatNums = scores(indx).PatNums;
% thesescores = scores(indx).scores;
% biopsyresults = scores(indx).biopsyresults;

T = struct2table(thesescores)
Tsorted = sortrows(T, 'scores')

save('tolookattable.mat', 'T')

% To look at:



% INN_248
