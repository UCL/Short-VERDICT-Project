function [vol, m] = n2mat(ninfo, varargin)
% N2MAT Nifti to matrix and m.geom structure
% Default assumes Nifti file was written in row-wise order (non MATLAB source)
% Cannot handle files with qfactor = -1 or more than 3 dimensions.
% Ongoing confusion with regards to qfactor and comparing Niftis from
% various sources. Currently need to reverse slice order in one test from a
% Philips TRA Nifti.
%
% THIS SOFTWARE IS PROVIDED  "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
% POSSIBILITY OF SUCH DAMAGE.
%
% 
% [vol, m] = n2mat(ninfo)  
%
% [vol, m] = n2mat(ninfo, Name, Value,  ...)
%
% Name, Value pairs:
% 'op'
%   value can be {'dv'} or 'pv' where 'dv' has linear scaling applied 
%   and 'pv' gives the raw pixel values.
%
% 'rc_order'
%   {'row'} | 'col'  
%   Data stored in file row-wise or column-wise. If NIFTI was written 
%   with MATLAB niftiwrite, likely to be in column order, otherwise in 
%   row order. Effect on image is a transpose. Note MATLAB dicomread/write
%   does perform a permutation of dims 1 and 2, but niftiread does not.
%
%  'reverse_slices'
%    'true' | {'false'}
%
% Example Usage:
%
%  ninfo = niftiinfo(filenm) ;
%  [vn, mn] = n2mat(ninfo, 'op','dv') ;
%
%  mn is structure with a geom structure (mn.geom)
%
% Note SliceThickness is slice centre separation (no information in NIFTI 
% about true slice width or slice gaps)
%
% Copyright David Atkinson, 2020, University College London
% D.Atkinson@ucl.ac.uk
%
% See also VRESAMPLE DICOM2INTRINSIC D2MAT NIFTIINFO
%

op = 'dv' ;
rc_order = 'row' ;
reverse_slices = 'false' ;

for ipv = 1:2:length(varargin)
    val = varargin{ipv+1} ;
    switch varargin{ipv}
        case 'op'
            op = val ;
        case 'order'
            rc_order = val ;
        case 'reverse_slices'
            reverse_slices = val ;
        otherwise
            error('Unknown parameter')      
    end
end

if ninfo.Qfactor ~= 1
    error(['Currently can only handle qfactor = 1'])
end

vol = niftiread(ninfo) ;
switch op
    case 'dv'
        vol = double(vol)*ninfo.MultiplicativeScaling + ninfo.AdditiveOffset ;
    case 'pv'
        % do nothing
    otherwise
        error('op can only be dv or pv')
end

szvol = size(vol) ;

switch rc_order
    case 'row'
        vol = permute(vol, [2 1 3:length(szvol)]) ; % permute as MATLAB niftiread just
        % reads bytes directly into array
        % Problematic as MATLAB uses column order but C uses row.
        % For NIFTI's written by non-MATLAB code, use 'row'...?
        szvol = size(vol) ;
    case 'col'
        % Do nothing
    otherwise
        error('unknown order')
end

if ninfo.raw.dim(1) > 3
    error(['Not implemented for ndims > 3'])
end

if reverse_slices == true
    warning('Will REVERSE SLICES')
    vol = flip(vol,3) ; 
end


% Set geometry parameters

% Determine affine matrix to convert intrinsic coordinates to LPH.
% Note NIFTI uses RAS

Anifti = ninfo.Transform.T ;

Tsub1 =[ 1  0  0  0 ; 
         0  1  0  0 ; 
         0  0  1  0 ; 
        -1 -1 -1  1 ]; % translation of -[1,1,1]
    
Tras2lph = diag([-1 -1 1 1]) ; 

% note seems to be that implied intrinsic is the same as MATLAB's.

A = Tsub1 * Anifti * Tras2lph ; % intrinsic to LPH

% Set IOP and PixelSpacing which should be constant
ciIPP = [ 1 1 1 1] ;  % Homogeneous intrinsic coordinate for first slice
clphIPP = ciIPP * A ; % Homogeneous LPH coordinate for first slice
    
row = ([2 1 1 1] * A ) - clphIPP ; % vector along a row
row(4) = [] ;
PixelSpacing_HW(2,1) = norm(row) ;
IOP(1:3) = row/norm(row) ;

col = ([1 2 1 1] * A ) - clphIPP ; % vector down a column
col(4) = [] ;
PixelSpacing_HW(1,1) = norm(col) ;
IOP(4:6) =  col/norm(col) ;

for islice = 1: szvol(3)
    ci11 = [ 1 1 islice 1] ; % Homogeneous intrinsic coordinate for slice
    clph11 = ci11 * A ; 
    
    
    row = ([2 1 islice 1] * A ) - clph11 ;
    row(4) = [] ;
    ps_hw(2) = norm(row) ;
    iop123 = row/norm(row) ;

    col = ([1 2 islice 1] * A ) - clph11 ;
    col(4) = [] ;
    ps_hw(1) = norm(col) ;
    iop456 = col/norm(col) ;
    
    if norm(ps_hw - PixelSpacing_HW) > 0.01
        error(['Pixel Spacing Issue'])
    end
    
    if (norm(iop123-IOP(1:3)) > 0.01) || (norm(iop456-IOP(4:6)) > 0.01)
        error(['IOP issue'])
    end
    
    % Prior to reverse_slices, the first slice has IPP of the last   
    if reverse_slices == true
        geom(islice,1).IPP =  clph11(1:3)' - ...
            (szvol(3)-1)*ninfo.PixelDimensions(3)*cross(IOP(1:3), IOP(4:6))' ; 
    else
        geom(islice,1).IPP = clph11(1:3)' ;
    end
    geom(islice,1).IOP = IOP' ;
    geom(islice,1).PixelSpacing_HW = PixelSpacing_HW ;
    geom(islice,1).Height = szvol(1) ;
    geom(islice,1).Width  = szvol(2) ;
    % SliceThickness in Nifti pixdim is slice centre separation
    geom(islice,1).SliceThickness = ninfo.PixelDimensions(3) ;
    geom(islice,1).source = 'NIFTI' ;
end

m.geom = geom ;


