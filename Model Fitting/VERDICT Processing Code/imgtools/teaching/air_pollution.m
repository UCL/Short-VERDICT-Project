        %
%  51.527975000000, -0.128774200000
%  https://www.londonair.org.uk/london/asp/datasite.asp?CBXSpecies2=NO2m&day1=1&month1=jan&year1=2013&day2=1&month2=jan&year2=2020&period=daily&ratidate=&site=CD9&res=6&Submit=Replot+graph
lat = 51.527975000000 ;
long = -0.128774200000 ;

webmap('Open Street Map')
wmcenter(lat, long, 10)
wmmarker(lat,long)
