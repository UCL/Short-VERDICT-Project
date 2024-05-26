function show_20171116


disp(['Enter folder DICOMs'])
pn = pref_uigetdir('show_20171116','pn') ;

slc = 3 ;

itype = 5;

fns = [1 4 5] ;

b0fn = 'IM_0064_WIP_DAB0mapWFS2_CLEAR_2901' ;
[vB0, mB0] = B0read(fullfile(pn,b0fn)) ;

fn{1} = 'IM_0017_WIP_0_30_IFA8_6s_SENSE_801' ;
txt{1} = set_txt(fn{1}, pn) ;
ls{1} = '-' ; lw(1) = 2;

fn{2} = 'IM_0035_WIP_300_30_IFA8_6s_SENSE_1501' ; 
txt{2} = set_txt(fn{2}, pn) ;
ls{2} = '--' ; lw(2) = 2;

fn{3} = 'IM_0037_WIP_600_30_IFA8_6s_SENSE_1601' ; 
txt{3} = set_txt(fn{3}, pn) ;
ls{3} = ':' ; lw(3) = 2;

fn{4} = 'IM_0039_WIP_300_30_IFA8_6s_SENSE_1701' ; 
txt{4}= set_txt(fn{4}, pn) ; 
ls{4} = '--' ; lw(4) = 2;

fn{5}  = 'IM_0040_WIP_600_30_IFA8_6s_SENSE_1801' ; 
txt{5} = set_txt(fn{5}, pn) ;
ls{5} = ':' ; lw(5) = 2;


coord{1} = [25 83];  ft{1} = 'pH7.25  354ms -16Hz' ;
coord{2} = [32 39];  ft{2} = 'pH6.86  282ms -11Hz' ;
coord{3} = [60 84];  ft{3} = 'pH6.49  298ms +3Hz' ;
coord{4} = [67 40] ; ft{4} = 'pH6.17  386ms -13Hz' ;
coord{5} = [96 86] ; ft{5} = 'pH5.86  446ms +9Hz' ;
coord{6} = [102 41];  ft{6} = 'PBS  1680ms -25Hz' ;

coords = [6 1 4 2 3 5];

% Looking at relation to B0 offset
dinfo = datparse(fullfile(pn,fn{1})) ; % No Sat
[vno, mno] = d2mat(dinfo,{'slice','ctdt','itype'},'itype',itype,'op','fp') ;
%%dinfo = datparse(fullfile(pn,fn{3})) ; % 
%%[v600,m600] = d2mat(dinfo,{'slice','ctdt','itype'},'itype',itype,'op','fp') ;

dinfo = datparse(fullfile(pn,fn{5})) ; % 
[v300,m300] = d2mat(dinfo,{'slice','ctdt','itype'},'itype',itype,'op','fp') ;

vBr = dreslice(vB0,mB0,mno) ;
 
b0s = vBr(:,:,slc) ;
sats = sum(vno(:,:,slc,end-3:end),4)/4 - sum(v300(:,:,slc,end-3:end),4)/4;
eshow(sats)
eshow(-b0s,'Name','-b0s')
figure('Name','sat recov vs B0')
plot(b0s(:), sats(:),'x')


for ifn = 1:length(fns)
    dinfo = datparse(fullfile(pn,fn{fns(ifn)})) ;
    [v{ifn}, m{ifn}, locs] = d2mat(dinfo,{'slice','ctdt','itype'},'itype',itype,'op','fp') ;
    ctdt{ifn} = [dinfo(locs(slc,:)).CardiacTriggerDelayTime] ;
    eshow(v{ifn}(:,:,slc,:),'Name',fn{fns(ifn)})
end


col_order = colorder(6) ;

figure('Name','data') ;
hold on, grid on
iline = 1 ;    
for ic = 1:length(coords)
    for ifn = 1:length(fns)
        cs = coord{coords(ic)} ;
        sgn = squeeze(v{ifn}(cs(1), cs(2),slc,:)) + ...
              squeeze(v{ifn}(cs(1), cs(2),slc+1,:)) + ...
              squeeze(v{ifn}(cs(1), cs(2),slc-1,:)) ;
        plot(ctdt{ifn},sgn/3,'LineWidth',lw(fns(ifn)),'LineStyle',ls{fns(ifn)},'Color',col_order(ic,:))
        lgt{iline} = [ft{coords(ic)} ,' ',txt{fns(ifn)}] ; 
        iline = iline + 1;
    end
    
end
legend(lgt)
axis([0 6000 0 7e6])

end

function txt = set_txt(imfn, pn)

dl = dir(fullfile(pn,['XX*',imfn(8:end)] )) ;
xxfn = dl.name ;

xxinfo = XXread(fullfile(pn,xxfn)) ;
txt = [num2str(xxinfo.EX_REPP_freq_offset), 'Hz ', ...
           num2str(xxinfo.EX_REPP_pulse_angles(1)),'deg'] ; 

end
