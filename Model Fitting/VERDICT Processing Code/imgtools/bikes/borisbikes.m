function borisbikes
% BORISBIKES Display docking station information from TFL API
%
% David Atkinson  D.Atkinson@ucl.ac.uk

viewall = false ;

if viewall
    % Gets all information. Puts name and id in a file for searching
    bp_all_url = 'https://api.tfl.gov.uk/BikePoint' ;
    
    ball = webread(bp_all_url) ;
    
    nall = length(ball) ;
    
    fn = tempname ;
    fid = fopen(fn,'w') ;
    
    for iball = 1:nall
        str = [ball(iball).id,' ',ball(iball).commonName];
        fprintf(fid,'%s\n',str) ;
    end
    fclose(fid) ;
    edit(fn)
end


% 

%BikePoints 
bp = [ ...
    329 ;
    271 ;
    343 ;
    7   ;
    247 ;
    239 ;
    98 ;
    65 ;
    12 ;
    20 ;
    795 ;
    ];

options = weboptions('Timeout', 10) ;
for ibp = 1:length(bp)
    bp_url = ['https://api.tfl.gov.uk/BikePoint/BikePoints_',num2str(bp(ibp))] ;
    try
        bpd = webread(bp_url, options) ;
        bpda = bpd.additionalProperties ;
        % extract modified time (in Zulu) and correct for summer time.
        C = textscan(bpda(8).modified, '%d %d %d %d %d %f', 'Delimiter','-:ZT') ;
        t = datetime(C{1:6},'Timezone','Europe/London') ;
        dt = tzoffset(t) ;
        tstr = datestr(t+dt, 'HH:MM:SS') ;
        
        str = sprintf('%+45s  B:%3s D:%3s (%s)',bpd.commonName,bpda(7).value, ...
            bpda(8).value, tstr) ;
    catch ME
        str = ME.message ;
    end
    disp(str)
end
