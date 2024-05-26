% d2tshow
%d2tshow Display tensors from DICOM
%   
% Will work with MultiFrame DICOM and PAR/REC. PAR/REC is much faster.
%
% When figure first appears, advance one plane on tshow control window to bring
% up a slice.
% For image display, select FA colour coded l1 for colour coded FA
% See help for tshow_keyp for all options
% To display helical angle, select one slice, click at apex and press 'a'
%  then select another point and click 'b' for base (sets long axis).
%  Helical and inclination angle colouring can then be chosen from Image
%  Display drop down.
%
% From the main figure menu, you can go to View and select the camera
% toolbar. This enables you to zoom and rotate the figure. Note the buttons
% should not be active when trying to select a point on the figure for
% tracking/ellipsoid drawing/selecting long axis etc
%
% See also TSHOW_KEYP  TSHOW




dinfo = datparse ; % read in DICOM or PAR/REC files

% load data into matrices using d2mat.
[B0,  outpb0] = d2mat(dinfo,{'slice','bv'},'bv',0,'op','fp') ;

% check on bgrads (if data is from PAR, then there may be a final image
% that is a computed iso and should not be read in here for DTI).
%

bgrads_all = outpb0.bgrads ;
if norm(bgrads_all(end,:)) < 0.9
    bgrads_all = bgrads_all(1:end-1,:) ;
end

[DWI, outpdwi] = d2mat(dinfo,{'slice','bdirec'}, ...
                             'bdirec',[1:size(bgrads_all)], 'op','fp') ;


% Convert to tensors.
% If B0 is actually nont zero diffusion, it can be merged into DWI - see
% dwi2tensor
[D ] = dwi2tensor(DWI, outpdwi.bgrads, outpb0.bvVec_indata(end), B0) ;

% %JH colormap
% cfa=create_colour_FA(D);
% keyboard

% generate roation matrices for tshow
ipp = dinfo(1).ImagePositionPatient ;
iop = dinfo(1).ImageOrientationPatient ;
ps_rcs = outpb0.vdims ;

[rcs2xyz, xyz2rcs, Rg2rcs] = gen_dicom_mat(ipp, iop, ps_rcs)

% call tshow
tshow(D, B0, xyz2rcs,rcs2xyz,Rg2rcs)





