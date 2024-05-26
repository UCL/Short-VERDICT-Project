function [rcs2xyz, xyz2rcs, Rg2rcs] = gen_dicom_mat(ipp, iop, ps_rcs)
% GEN_DICOM_MAT Generate matrices to convert between DICOM LPS coords and
% MATLAB row,column, slice
% [rcs2xyz, xyz2rcs, Rg2rcs] = gen_dicom_mat(ipp, iop, ps_rcs)
%
% ipp [nslice 3] ImagePositionPatient (LPH coord of top left pixel). Uses
% only slice 1.
% iop [1 6] ImageOrientationPatient (LPH) unit vector ALONG a row, then
% along a column.
% ps_rcs [1 3] row spacing, column spacing, slice spacing.
%
% rcs2xyz [ 4 x 4] 
% xyz2rcs [ 4 x 4]
% Rg2rcs  [3 x 3]  converts a vector in LPH (e.g. an eigenvector) to row, column
%                  slice dirction.
%
% David Atkinson  D.Atkinson@ucl.ac.uk
%
% $Id: gen_dicom_mat.m 207 2008-10-11 21:23:23Z ucacdat $
%

XC=1;YC=2;ZC=3 ; % 
R=1;C=2;S=3;

% ImageOrientationPatient is;
%  row_dircos_x
%  row_dircos_y
%  row_dircos_z
%  col_dircos_x
%  col_dircos_y
%  col_dircos_z
%
% i.e. the vector ALONG the rows, then along the COLUMNS. NOT the direction
% of the rows and the ethe direction of the columns!
%
% The pixel spacing is in the order row_spacing then column_spacing
%
% So, if you go down by 2 rows, you travel in the direction of the column
% direction cosine and the distance is 2*row_spacing
% maybe .....
%

rdc = iop(1:3) ;
cdc = iop(4:6) ;

rdc = rdc ./norm(rdc) ;
cdc = cdc ./norm(cdc) ;
sdc = cross(rdc,cdc) ; % to give RH coord system ? ....

nslice = size(ipp,1) ;


islice = 1 ; % all relative to slice 1
rcs2origin = eye(4) ; rcs2origin(1:3,4) = [-1 ; -1; -1] ;
origin2xyz = eye(4) ; origin2xyz(1:3,4) =  ipp(islice,:)'  ;

rcs2xyzR = [cdc(XC)*ps_rcs(R)  rdc(XC)*ps_rcs(C)  sdc(XC)*ps_rcs(S) 0 ;
            cdc(YC)*ps_rcs(R)  rdc(YC)*ps_rcs(C)  sdc(YC)*ps_rcs(S) 0 ;
            cdc(ZC)*ps_rcs(R)  rdc(ZC)*ps_rcs(C)  sdc(ZC)*ps_rcs(S) 0 ;
             0                 0                 0                1 ] ;
       
rcs2xyz = origin2xyz * rcs2xyzR * rcs2origin ;
    
origin2rcs = eye(4) ; origin2rcs(1:3,4) = [1 ; 1; 1] ;
xyz2origin = eye(4) ; xyz2origin(1:3,4) = -ipp(islice,:)' ;

xyz2rcs = origin2rcs * inv(rcs2xyzR) * xyz2origin ;

Rg2rcs = inv(rcs2xyzR(1:3,1:3)) ;
