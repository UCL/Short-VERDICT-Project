function show_20171101


disp(['Enter folder DICOMs'])
pn = pref_uigetdir('show_20171101','pn') ;

slc = 4 ;

itype = 5;
 fn{1} = 'IM_0005_WIP_0_30_IFA8_SENSE_301' ; lg{1} = '0 30 IFA8 short TR' ;
% fn{2} = 'IM_0009_WIP_0_30_IFA15_SENSE_401' ; 
% fn{3} = 'IM_0011_WIP_0_30_IFA25_SENSE_501' ;
 fn{2} = 'IM_0013_WIP_250_30_IFA8_SENSE_601' ; lg{2} = '250 30 IFA8  short TR' ;
 fn{3} = 'IM_0015_WIP_470_30_IFA8_SENSE_701' ; lg{3} = '470 30 IFA8  short TR' ;

%fn{1} = 'IM_0017_WIP_0_30_IFA8_SENSE_801' ; lg{1} = '0 30 longTR' ;
%fn{2} = 'IM_0023_WIP_250_30_IFA8_SENSE_1101' ; lg{2} = '250 30 longTR';
%fn{3} = 'IM_0025_WIP_600_30_IFA8_SENSE_1201' ; lg{3} = '600 30 longTR' ;


for ifn = 1:length(fn)
    dinfo = datparse(fullfile(pn,fn{ifn})) ;
    [v{ifn}, m{ifn}, locs] = d2mat(dinfo,{'slice','ctdt','itype'},'itype',itype,'op','fp') ;
    ctdt{ifn} = [dinfo(locs(slc,:)).CardiacTriggerDelayTime] ;
    %eshow(v{ifn}(:,:,slc,:),'Name',escunder(fn{ifn}))
end


coord{1} = [74 60]; lgc{1} = 'water' ;
coord{2} = [70 46]; lgc{2} = '1 (8oclock)' ;
coord{3} = [52 52]; lgc{3} = '2' ;
coord{4} = [49 70] ; lgc{4} = '3' ;
coord{5} = [63 82] ; lgc{5} = '4' ;
coord{6} = [81 74]; lgc{6} = '5' ;

ls = {'-', '--', ':'} ;
col_order = colorder(6) ;

figure('Name','data') ;
hold on, grid on
iline = 1 ;    
for ic = 1:length(coord)
    for ifn = 1:length(fn)
        sgn = squeeze(v{ifn}(coord{ic}(1), coord{ic}(2),slc,:)) + ...
              squeeze(v{ifn}(coord{ic}(1), coord{ic}(2),slc+1,:)) + ...
              squeeze(v{ifn}(coord{ic}(1), coord{ic}(2),slc-1,:)) ;
        plot(ctdt{ifn},sgn/3,'LineWidth',2,'LineStyle',ls{ifn},'Color',col_order(ic,:))
        lgt{iline} = [lgc{ic} ,' ',lg{ifn}] ; 
        iline = iline + 1;
    end
    
end
legend(lgt)
axis([0 5000 0 3.75e7])


