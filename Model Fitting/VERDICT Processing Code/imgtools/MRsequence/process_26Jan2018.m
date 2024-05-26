% Prostate Patient 26 Jan 2018
% Testing of variations to DCE sequence parameters.
%
% Overall, little difference, perhaps more stability in ProSet 

folder = pref_uigetdir('process_26Jan2018','folder') ;

fnclin = 'I66_sDYNAMIC_15_816' ;
fnhl = 'I67_WIP_DCE_high-low_SENSE_901' ;
fnps121 = 'I69_WIP_DCE_proset121_SENSE_1001' ;

dclin = datparse(fullfile(folder,fnclin)) ;
[vclin, mclin] = d2mat(dclin,{'slice'},'op','fp') ;

dhl = datparse(fullfile(folder,fnhl)) ;
[vhl, mhl] = d2mat(dhl,{'slice'},'op','fp') ;

dps121 = datparse(fullfile(folder,fnps121)) ;
[vps121, mps121] = d2mat(dps121,{'slice'},'op','fp') ;

vall = cat(2,vclin/3e6,vhl/1.4e6,vps121/1.5e6) ;
eshow(vall)

