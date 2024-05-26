function [A = nifti2intrinsic(ninfo)
% NIFTI2INTRINSIC Calculate affine matrix transforming NIFTI to intrinsic space
%
% A = nifti2intrinsic()
%
% Example usage:
%
% Copyright David Atkinson, 2020, University College London
% D.Atkinson@ucl.ac.uk
%
% See also VRESAMPLE DICOM2INTRINSIC

% Create matrices that:
%   nifti-intrinsic to RAS
%
%   RAS to LPH  
%  diag([-1 -1 1 1])
%
%   MATLAB-intrinsic to nift-intrinsic  NOT NEEDED!
%     NIFTI intrinsic to MATLAB intrinsic is rotation about x 
%     by 180 degrees, translation by adding  1,1,1. So,
%     MATLAB-intrinsic to mifti-intrinsic is
%     Subtract [1,1,1], rotate about x by 180 degrees: diag([1 -1 -1 1])
%        
%
% Tsub1 * Trotx * ntform * TRAS2LPH
%
% Can then determine IPP, IOP, PS by finding LPH coord of MATLAB-intrinsic
% [1,1,1] and along row and down column.
%
% Beware of negative qfactor. Can get IPP etc correct, BUT sme algorithms
% assume RHS.

Tsub1 =[ 1  0  0  0 ; 
         0  1  0  0 ; 
         0  0  1  0 ; 
        -1 -1 -1  1 ]; % translation of -[1,1,1]
    
% Trotx = diag([1 -1 -1 1]) ;

Tras2lph = diag([-1 -1 1 1]) ;

ntformA = ninfo.Transform.T ;

% A = Tsub1 * Trotx * ntformA * Tras2lph

A = Tsub1 * ntformA * Tras2lph

ci111 = [1 1 1 1] ;
clphipp = ci111 * A

row = ([2 1 1 1] * A ) - clphipp ;
norm(row)
iop123 = row/norm(row)

col = ([1 2 1 1] * A ) - clphipp ;
norm(col)
iop456 = col/norm(col)


% MATLAB's niftiinfo does this for qform (note transpose in affine3d):
% % i = self.header.pixdim(2);
% % j = self.header.pixdim(3);
% % k = qfactor * self.header.pixdim(4);
% % 
% % R = [a*a+b*b-c*c-d*d 2*b*c-2*a*d     2*b*d+2*a*c
% %     2*b*c+2*a*d     a*a+c*c-b*b-d*d 2*c*d-2*a*b
% %     2*b*d-2*a*c     2*c*d+2*a*b     a*a+d*d-c*c-b*b];
% % 
% % T = [self.header.qoffset_x; ...
% %     self.header.qoffset_y; ...
% %     self.header.qoffset_z];
% % R = R * diag([i j k]);
% % 
% % xform = affine3d([R T; zeros(1,3) 1]');
                    
     
% So this is essentially a rotation, scale and translation to offset
%  with the following assumptions on the implied intrinsic:
%    origin at tlc, y up?


ntform = ninfo.Transform ;

% nifti intrinsic to RAS is returned by MATLAB above
% RAS to LPH is:
%
%  diag([-1 -1 0 1])
%
% NIFTI intrinsic to MATLAB intrinsic (needed?) is:
% rotation about x by 180 degrees, translation by adding  1,1,1

% how to get geom from nifti?

PixelSpacing_HW = [ ninfo.raw.pixdim(3) ninfo.raw.pixdim(2) ] ;
Height = ninfo.raw.dim(3) ;
Width = ninfo.raw.dim(2) ;

iop(1:3) =  [ -1*(a*a+b*b-c*c-d*d) -1*(2*b*c-2*a*d)     2*b*d+2*a*c ] ;
iop(4:6) = -[ -1*(2*b*c+2*a*d)     -1*(a*a+c*c-b*b-d*d) 2*c*d-2*a*b ] ;

ipp = [ -ninfo.raw.qoffset_x -ninfo.raw.qoffset_y ninfo.raw.qoffset_z ] - ...
    (Height-1)*PixelSpacing_HW(1)*iop(4:6) ; 


% Nifti is RAS+, i.e. -L, -P, S

A = Torig *  Trim * Tscale * T111 ; 


[slsep, rdc, cdc, sdc] = geom_check(geom) ;
psh = geom(1).PixelSpacing_HW(1) ;
psw = geom(1).PixelSpacing_HW(2) ;

origin_pp = geom(1).IPP ;
origin_pp = origin_pp(:)' ;  % row vector
   
Torig = [1 0 0 0 ; 
         0 1 0 0 ; 
         0 0 1 0 ; 
      -origin_pp 1] ;
  
T111 = [1 0 0 0 ; 
        0 1 0 0 ; 
        0 0 1 0 ; 
        1 1 1 1 ]; % translation of 1,1,1
    
Trim = [rdc(1) cdc(1) sdc(1) 0 ;
        rdc(2) cdc(2) sdc(2) 0 ;
        rdc(3) cdc(3) sdc(3) 0 ;
         0      0       0    1 ];
      
        
Tscale = [1/psw   0      0     0 ; 
          0     1/psh    0     0 ; 
          0       0   1/slsep  0 ; 
          0       0      0     1] ;
    
A = Torig *  Trim * Tscale * T111 ;   

% Check transform at some corner locations

thresh = 0.001 ; % threshold for vector difference

% top left centre
actv = [origin_pp 1] * A ;
if norm(actv(1:3) - [ 1 1 1 ]) > thresh
    actv
    error('TLC incorrect')
end

width = geom(1).Width ;
height = geom(1).Height ;

% bottom-right corner, this slice
brc = origin_pp + (rdc(:)' * psw * (width-1)) + (cdc(:)' * psh * (height-1)) ;
int_brc = [brc 1] * A ;

if norm(int_brc(1:3) - [ width height 1]) > thresh
    int_brc
    error('bottom right corner not correct')
end

% check corner opposite tlc, last slice
nslice = length(geom) ;

fc = brc + sdc(:)' * (nslice-1) * slsep ;

int_fc = [fc 1] * A ;

if norm(int_fc(1:3) - [width height nslice]) > thresh
    int_fc
    error(['far corner not correct'])
end




