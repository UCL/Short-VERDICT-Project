% John Hopkins data
%
% Problematic because days are like column heders - need to read in twice
% once for top row a datetime
% Will sort out US day/month order and 2020
pn = '/Users/davidatkinson/OneDrive - University College London/teaching/DoMcomputing/COVID-19-master/csse_covid_19_data/csse_covid_19_time_series' ;
fn = '/time_series_covid19_deaths_global.csv' ;

ffn = fullfile(pn,fn) ;

opts = detectImportOptions(ffn) ;

opts1 = setvartype(opts,'datetime') ;
opts1 = setvaropts(opts1,'InputFormat','MM/dd/yy') ; % need this to prevent abiguity over dd/mm, mm/dd

tsd = readtable(ffn, opts1) ;

ds = tsd{1,5:end} ;


% % opts.DataLines = [2 Inf] ;
% % opts.VariableNamesLine = 1 ;
tsdc = readtable(ffn,opts) ;

UKrow = 225 ;  % Note row is 225 if top line is included
% % tsdc.Country_Region(UKrow) 

deaths_UK = tsdc{UKrow,5:end} ;

plot(ds, deaths_UK)
grid on

