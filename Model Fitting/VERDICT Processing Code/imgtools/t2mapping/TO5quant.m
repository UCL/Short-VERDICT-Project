function TO5quant(gel_order)
% TO5quant Quantitative imaging on TO5 phantom
%   Assumes IR T1 measures and a multi-echo with scanner calc T2
%
%  TO5quant(gel_order)
%    gel_order is order of gels in phantom slots, from top left, 
%    along rows e.g. 
%      gel_order = [1 2 3 5 6 7 8 10 12 13 15 17];  
%
%  
%   Sept 14, 2020  gel_order = [1 2 3   5 6 7 8   10    12 13    15   17    ] ; 
%   Sept 22, 2020  gel_order = [1 2 3 4       8 9 10 11       14 15 16    18] ;
%
% David Atkinson  D.Atkinson@ucl.ac.uk
%
% See also TO5findcircles relax3T
%

disp(['Select all Inversion Recovery T1s, then ME T2'])
fns_t1 = dselect('message','Multi-select IR T1s', 'prefname','TO5IRT1') ; 
fn_t2  = dselect('message','Select ME T2','prefname','TO5T2') ;

% T1
dinfoIR = datparse(fns_t1) ;  
ITYPE = 6 ;  % 6 for magnitude, 8 for real (don't use real) 
[vir, mir] = d2mat(dinfoIR,{'InversionTime','itype'},'itype',ITYPE,'op','fp') ; 
opt = 'lsqnl_mag3' ; % or 'lsqnl_mag2' or 'RD-NLS-PR3'
fitt1 = IRT1(vir, mir.tiVec, opt) ;

% T2
[T2, figname, vet2, met2, S0] = MET2(fn_t2) ;

% Find gels in IR
[roist1, roift1] = TO5findcircles(sum(vir,3), mir) ;

% Find gels in ME
[roist2, roift2] = TO5findcircles(vet2(:,:,1,1), met2) ;

slabels = ['ABCDEFGHIJKL'] ; % slot labels

figure('Name','T1 signals')
haxt1 = gca ; hold on ; grid on
figure('Name','T2 signals')
haxt2 = gca ; hold on ; grid on

for islot = 1:length(slabels)
    gel = gel_order(islot) ;
    
    % Identify the roi number corresponding to this slot. 
    % rois have labels with the slot as a letter
    
    jroit1 = slot2roi(islot, roist1, slabels) ;
    jroit2 = slot2roi(islot, roist2, slabels) ;
    
    bw = createMask(roist1{jroit1}) ;
    valst1 = fitt1.T1(bw) ;
    
    bw = createMask(roist2{jroit2}) ;
    valst2 = T2(bw) ;
    
    prop.tempC = 20 ;
    prop.gel = gel ;
    R = relax3T('TO5',prop) ;
    
    fmt = '%7.2f' ;
    disp(['Gel ',num2str(gel,'%02d'), ' Mean T1: ',num2str(mean(valst1(:)),fmt), ...
        ' ref T1: ',num2str(R.T1,fmt), ...
        ':: Mean T2: ',num2str(mean(valst2(:)),fmt),' ref T2: ',num2str(R.T2,fmt)])

    % visualise
    
    cent = roist1{jroit1}.Center ;
    cr = round(cent(2)); cc = round(cent(1)) ;
    hp1 = plot(haxt1, mir.tiVec, squeeze(vir(cr,cc,:)),'o-', ...
        'LineWidth',2,...
        'DisplayName',['Gel ',num2str(gel),' T1 ',num2str(fitt1.T1(cr,cc))]) ;
    calcIT = [0:10:max(mir.tiVec)] ;
    sig = abs( fitt1.M0(cr,cc) * (1-2*fitt1.RB(cr,cc)*exp(-calcIT/fitt1.T1(cr,cc)))) ;
    plot(haxt1, calcIT, sig, '--', 'Color',hp1.Color, 'LineWidth',2)
    
    cent = roist2{jroit2}.Center ;
    cr = round(cent(2)); cc = round(cent(1)) ;
    hp2 = plot(haxt2, met2.effTE, squeeze(vet2(cr, cc,1,:)),...
        'LineWidth',2, ...
        'DisplayName',['Gel ',num2str(gel),' T2 ',num2str(T2(cr,cc))]) ;
    calcTE = [0:max(met2.effTE)] ;
    sig = S0(cr,cc)*exp(-calcTE/T2(cr,cc));
    plot(haxt2, calcTE, sig,'--','Color',hp2.Color, 'LineWidth',2)

end
legend(haxt2)
legend(haxt1)


end

function [jroi] = slot2roi(islot, rois, slabels)
% slot2roi Slot number to ROI number

jroi = [] ;
for iroi = 1:length(rois)
    if strcmp(slabels(islot), rois{iroi}.Label)
        jroi = iroi ;
    end
end

if isempty(jroi)
    warning('No ROI found')
end

end
