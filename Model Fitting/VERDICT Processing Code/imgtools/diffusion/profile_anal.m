function profile_anal
% PROFILE_ANAL Analysis of ADC profiles for SI Joints
% Assumes .mat files in one folder, filenames for patients start 'case' and
% for controls start 'cont'. .mat files should contain an roistats 
% structure and a profile labelled either 'N','normal', or 'Normal' 
% to be teeated as the reference.
% Default is to use the average of the whole Normal profile as the ADC
% threshold.
% Also, will ignore end slices by default.
%
% Default ADC threshold is mean ADC along whole line of Normal.
%
% David Atkinson  D.Atkinson@ucl.ac.uk
%
% See also ROIANAL

% Select folder with .mat files
% Enter fixed line length and threshold
% For each file:
%  Read roistats
%  Check line length is longer than fixed length
% Truncate lines
%  Check no ADC is below threshold
% Calculate areas above threshold over fixed length
% Identify profile for normal if possible
% Report all areas above threshold,
%   if normal (reference) identified, report differences and average if L
%   and R identified.
% Plot profiles, labels and data in legend.
% Generate report?


h_old = findobj('Tag','profile_anal') ;
if ~isempty(h_old)
    warning(['Old profile_anal figures will be deleted.'])
end

if ispref('profile_anal','infolder')
    infolder = getpref('profile_anal','infolder') ;
else
    infolder = [] ;
end

infolder = uigetdir(infolder,'Select folder containing .mat results files') ;
if infolder == 0
    return
else
    setpref('profile_anal','infolder',infolder) 
end

prompt = {'Enter profile length:','Enter ADC threshold (N mean normal):','Ignore end slices Y/N:'};
dlg_title = 'Input for profile analysis function';
num_lines = 1;
def = {'14','N','Y'};  % default length and ADC
answer = inputdlg(prompt,dlg_title,num_lines,def);

[plen status1] = str2num(answer{1}); 
[tADC status2] = str2num(answer{2}) ;
ig_end = answer{3} ;

if strcmpi(ig_end,'Y')
    ignore_end = true ;
else
    ignore_end = false ;
end

if isempty(tADC)
    thresh_meth = 'normal_mean' ;
else
    thresh_meth = 'user_thresh' ;
end


fmat = dir(fullfile(infolder,'*.mat')) ;
nfiles = length(fmat) ;
disp([num2str(nfiles),' .mat files found in: ',infolder])

switch thresh_meth
    case 'normal_mean'
        disp(['Parsing for normal file to obtain average ADC as suggested threshold'])
        NtADC = zeros([1 nfiles]) ;
        
        for ifile = 1:nfiles
            [pmat, fnmat, extmat] = fileparts(fmat(ifile).name) ;
            
            S = load(fullfile(infolder,fmat(ifile).name)) ;
            roiexp = [S.roistats{:}] ; % expanded cell array
            llens = [roiexp.llen] ; % column vector of linelengths drawn
            
            % Find the reference line labelled 'N', 'Normal' or 'normal'
            lsc = [S.labstrc{:}];
            tf = strcmp(lsc,'N') + strcmp(lsc,'Normal') + strcmp(lsc,'normal') ;
            if sum(tf)~=1
                warning(['Found: ',num2str(sum(tf)),' labels ', ...
                    ' expecting 1.'])
                disp([' Labels were: ',lsc])
                iref = [] ;
            else
                iref = find(tf==1) ;
                % get (full) line profile, average and set to defADC
                
                dists = roiexp(iref).dist ;
                norm_len = dists(end)-dists(1) ;
                np = round(4*norm_len) ;
                newx = linspace(-norm_len/2, norm_len/2, np) ;
                
                newrefprof = interp1(roiexp(iref).dist, roiexp(iref).profile, newx,...
                    'pchip','extrap') ;
                
                NtADC(ifile) = mean(newrefprof) ;
            end
        end
    case 'user_thresh'
        NtADC = repmat(tADC,[1 nfiles]) ;
end



if ~isempty(h_old)
    close(h_old)
end

figure('Name', 'All values','Tag','profile_anal') ;
hold on
hascat = gca ;

itable = 1 ;
profile_anal_table{itable, 1} = 'Label & slice' ;
profile_anal_table{itable, 2} = 'Area-Aref' ;
profile_anal_table{itable, 3} = 'mean_profile-mean_ref' ;
profile_anal_table{itable, 4} = 'mean_profile' ;
        
for ifile = 1:nfiles
    disp(' ')
    [pmat, fnmat, extmat] = fileparts(fmat(ifile).name) ;
    tADC = NtADC(ifile) ;
    disp(['Reset threshold ADC'])
    disp([fnmat, ' , threshold ',num2str(tADC)]) % e.g. for cont16.mat, outputs cont16
    
    %disp(['Processing File: ',fmat(ifile).name])
    itable = itable + 1 ;
    profile_anal_table{itable, 1} = fnmat ;
    
    cont = true ; this_col = [0 0 1] ; this_mk = '+' ; this_shift = +0.05 ;
    if ~isempty(findstr('case',fmat(ifile).name))
        cont = false; this_col = [1 0 0] ; this_mk = '+' ; this_shift = -0.05 ;
    end
    
    S = load(fullfile(infolder,fmat(ifile).name)) ;
    
    
    roiexp = [S.roistats{:}] ; % expanded cell array
    llens = [roiexp.llen] ; % column vector of linelengths drawn
    
    if min(llens) < plen
        warning(['The shortest profile (',num2str(min(llens)), ...
            ' is less than the fixed length of ',num2str(plen)])
    end
    
    % Find the reference line labelled 'N', 'Normal' or 'normal'
    lsc = [S.labstrc{:}];
    tf = strcmp(lsc,'N') + strcmp(lsc,'Normal') + strcmp(lsc,'normal') ;
    if sum(tf)~=1
        warning(['Found: ',num2str(sum(tf)),' labels ', ...
            ' expecting 1.'])
        disp([' Labels were: ',lsc])
        iref = [] ;
    else
        iref = find(tf==1) ;
    end
    
    slcs = S.slice ;
    slcs_noref = slcs ;
    slcs_noref(iref) = [] ;
    uq_slcs = unique(slcs_noref) ;
    st_sl = uq_slcs(1) ;
    fn_sl = uq_slcs(end) ;
    
    
    % truncate line to fixed length plen
    np = round(4*plen) ;
    newx = linspace(-plen/2, plen/2, np) ;
    dx = newx(2)-newx(1) ;
    
    newrefprof = interp1(roiexp(iref).dist, roiexp(iref).profile, newx,...
        'pchip','extrap') ;
    
    
    ct = newrefprof - tADC ;
    meanref = mean(newrefprof) ;
    
    ct(ct<0)=0 ;
    Aref = sum(ct)*dx ;
    
    figure('Name',fmat(ifile).name,'Tag','profile_anal') ;
    hold on
    haf = gca ;
    % plot(newx,newrefprof,'k'), hold on
    xlabel('Distance')
    ylabel('ADC')
    title([fmat(ifile).name,'. Thresh ADC: ',num2str(tADC),...
        '. Ref area above thresh: ',num2str(Aref)])
    nrow = 1 ;
    
    nprofile = length(lsc) ;
    col_order = colorder(nprofile) ;
    
    ileg_prof = 0 ;
    for iprofile = 1 : nprofile
        itable = itable + 1 ;
        newprofile = interp1(roiexp(iprofile).dist, ...
            roiexp(iprofile).profile, newx, 'pchip','extrap') ;
        ct = newprofile - tADC ;
        ct(ct<0) = 0 ;
        A = sum(ct)*dx ;
        
        label = [S.labstrc(iprofile)];
        outstr = [label{1}{:}, ' A-Aref: ',num2str(A-Aref,'%8.1f'), ...
            '. mn-ref: ', num2str(mean(newprofile)-meanref,'%7.1f'), ...
            '. mn: ', num2str(mean(newprofile),'%7.1f')] ;
        csvoutstr = [label{1}{:}, '  , ', num2str(A-Aref,'%8.1f') ];
            
        %disp(outstr)
        disp(csvoutstr)
        
        profile_anal_table{itable, 1} = [label{1}{:}, ' sl:',num2str(S.slice(iprofile))] ;
        profile_anal_table{itable, 2} = num2str(A-Aref,'%8.1f') ;
        profile_anal_table{itable, 3} = num2str(mean(newprofile)-meanref,'%7.1f') ;
        profile_anal_table{itable, 4} = num2str(mean(newprofile),'%7.1f');
        
        if iprofile ~= iref
            if ignore_end && (slcs(iprofile)==st_sl || slcs(iprofile)==fn_sl )
                % skip
            else
                ileg_prof = ileg_prof + 1 ;
                plot(haf, newx, newprofile,'LineWidth',2,'Color',col_order(nrow,:))
                nrow = nrow + 1;
                plot(hascat,1+this_shift,A-Aref,this_mk,'Color',this_col)
                plot(hascat,2+this_shift,5*(mean(newprofile)-meanref),this_mk,'Color',this_col)
                plot(hascat,3+this_shift,5*(mean(newprofile)),this_mk,'Color',this_col)
                leg{ileg_prof} = outstr ;
            end
        else
            ileg_prof = ileg_prof + 1 ;
            plot(haf, newx, newprofile,'LineWidth',4,'Color',[0 0 0])
            leg{ileg_prof} = outstr ;
        end
    end
    
    axis(haf,[-plen/2 plen/2 0 3500])
    line([-plen/2 plen/2],[tADC tADC],'Color',[1 0 0],'LineWidth',2,...
        'LineStyle',':')
    ileg_prof = ileg_prof + 1 ;
    leg{ileg_prof} = 'thresh' ;
    if nprofile < 10
        legend(haf,leg)
    end
    
    title(hascat,'Cases red, controls blue')
    
    %axis(hascat,[0 4 -1000 12000 ] ) ;
    set(hascat,'XTick',[1 2 3])
    xlabel(hascat,'Area difference [1], 5xmean difference [2], 5xmean [3]')
end

assignin('base', 'profile_anal_table', profile_anal_table) ;

if ispref('profile_anal','rep_file')
    deffn = getpref('profile_anal','rep_file');
else
    deffn = [] ;
end

% in future could offer choice of report format
disp(' ')
disp(['Enter Report filename in UI (may be hidden behind figures)'])
[fn,pn,FilterIndex] = uiputfile({'*.pdf','PDF report'},'Enter filename',deffn) ;

if fn == 0
    warning(['No file specified'])
    return
else
    ffn = fullfile(pn,fn) ;
    setpref('profile_anal','rep_file',ffn)
end

if exist('RptgenML.CReport','class')
    [RptgenML_CReport1] = buildprofile_anal ;
    disp(['Writing report to: ',ffn])
    report(RptgenML_CReport1,['-o',ffn],'-noview') ;
else
    disp(['No report generated - MATLAB report toolbox not present?'])
end
