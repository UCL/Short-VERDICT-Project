%function ismrmrd2dicom
% ISMRMRD2DICOM Convert from ISMRMRD flags to DICOM IOP, IPP
%
% Convert from the ISMRMRD format information to DICOM IOP, IPP.
% Aim is to understand Siemens mMR data for future link to PET geometry.
%
% To do:
%  Remove hardcoding of reconSpace parameters (fetch from xml data)
%  Check when RO and PE directions are swapped within a given orientation.
%  Check for angulated slices, SAG and TRA orientations.
%  Check for slices in opposite order and sequences where the fat shift
%  direction might be reversed.


% Needs automating:
nky = 256; nslice = 24;
reconSpace.matrixSize.x = 256 ;
reconSpace.matrixSize.y = 256 ;
reconSpace.fieldOfView.x = 420 ;
reconSpace.fieldOfView.y = 420 ;
PS_HW = [reconSpace.fieldOfView.y/reconSpace.matrixSize.y  ...
         reconSpace.fieldOfView.x/reconSpace.matrixSize.x ] ;

     
disp('Select H5 file')
[fn, dsn] = explore_h5 ;

data = h5read(fn,dsn{1 }) ;

read_dir = data.head.read_dir ;

[nr, naq] = size(read_dir) ;
if nr~=3 , error('position should have size 3 for first dimension'), end

phase_dir = data.head.phase_dir ;
slice_dir = data.head.slice_dir ;
position = data.head.position ;

% Check read, phase and slice directions are orthogonal:
drp = dot(read_dir, phase_dir, 1) ;
drs = dot(read_dir, slice_dir, 1) ;
dps = dot(phase_dir, slice_dir, 1) ;

if sum(drp) > naq*eps('single')
    warning([' Dot products for read, phase dirs not all zero'])
end
if sum(drs) > naq*eps('single')
    warning([' Dot products for read, slice dirs not all zero'])
end
if sum(dps) > naq*eps('single')
    warning([' Dot products for phase, slice dirs not all zero'])
end


LEFT = [1 0 0]; % LPH coordinates
POSTERIOR = [0 1 0];
FOOT = [0 0 -1] ;
HEAD = -FOOT ;
RIGHT = -LEFT;
ANTERIOR = -POSTERIOR ;

TOL = 1e-4; % tolerance on vector normalisation test
for iaq = 1 :nky:nky*nslice
    % Check normalised and direction
    rd = read_dir(:,iaq) ; pd = phase_dir(:,iaq) ; sd = slice_dir(:,iaq) ;
    cp = cross(rd,pd);
    if abs(norm(rd)-1) > TOL , warning('read_dir not normalised'), end
    if abs(norm(pd)-1) > TOL , warning('phase_dir not normalised'), end
    if abs(norm(sd)-1) > TOL , warning('slice_dir not normalised'), end
    if sum(abs(sd+cp)) < 3*eps('single')
        warning(['LEFT-HANDED?: slice_dir is negative cross product of read and phase'])
    end
    
    % Determine radiological orientation from plane normal
    dptra = dot(cp,cross(LEFT,POSTERIOR)) ; % dot product with normals
    dpcor = dot(cp,cross(LEFT,FOOT)) ;      % of radiological slices
    dpsag = dot(cp,cross(HEAD,POSTERIOR)) ;
        
    [mx, loc] = max([abs(dptra) abs(dpcor) abs(dpsag)]) ;
    switch loc
        case 1
            ori = 'TRA' ;
        case 2
            ori = 'COR' ;
        case 3
            ori = 'SAG' ;
    end
    
    disp(['Determined ORI: ',ori])
    
    % Set IOP
    % Find which nominal direction for the determined radiological 
    % orientation the read and phase directions are closest to. Assign to
    % IOP the relevant direction and sign.
    IOP = zeros([1 6]) ;
    switch ori
        case 'TRA'
            [mx, loc] = max([ dot(rd,LEFT) dot(rd,POSTERIOR) dot(rd,RIGHT) dot(rd, ANTERIOR)]) ;
            switch loc
                case 1
                    IOP(1:3) = rd;
                case 2
                    IOP(4:6) = rd ;
                case 3
                    IOP(1:3) = -rd ;
                case 4
                    IOP(4:6) = -rd ;
            end
            [mx, loc] = max([ dot(pd,LEFT) dot(pd,POSTERIOR) dot(pd,RIGHT) dot(pd, ANTERIOR)]) ;
            switch loc
                case 1
                    IOP(1:3) = pd;
                case 2
                    IOP(4:6) = pd ;
                case 3
                    IOP(1:3) = -pd ;
                case 4
                    IOP(4:6) = -pd ;
            end
    
    
        case 'COR'
            [mx, loc] = max([ dot(rd,LEFT) dot(rd,FOOT) dot(rd,RIGHT) dot(rd, HEAD)]) ;
            switch loc
                case 1
                    IOP(1:3) = rd;
                case 2
                    IOP(4:6) = rd ;
                case 3
                    IOP(1:3) = -rd ;
                case 4
                    IOP(4:6) = -rd ;
            end
            
            [mx, loc] = max([ dot(pd,LEFT) dot(pd,FOOT) dot(pd,RIGHT) dot(pd, HEAD)]) ;
            switch loc
                case 1
                    IOP(1:3) = pd;
                case 2
                    IOP(4:6) = pd ;
                case 3
                    IOP(1:3) = -pd ;
                case 4
                    IOP(4:6) = -pd ;
            end
            
         case 'SAG'
            [mx, loc] = max([ dot(rd,HEAD) dot(rd,POSTERIOR) dot(rd,FOOT) dot(rd, ANTERIOR)]) ;
            switch loc
                case 1
                    IOP(1:3) = rd;
                case 2
                    IOP(4:6) = rd ;
                case 3
                    IOP(1:3) = -rd ;
                case 4
                    IOP(4:6) = -rd ;
            end
            [mx, loc] = max([ dot(pd,HEAD) dot(pd,POSTERIOR) dot(pd,FOOT) dot(pd, ANTERIOR)]) ;
            switch loc
                case 1
                    IOP(1:3) = pd;
                case 2
                    IOP(4:6) = pd ;
                case 3
                    IOP(1:3) = -pd ;
                case 4
                    IOP(4:6) = -pd ;
            end
    end
    
    if sum(IOP(1:3))==0 || sum(IOP(4:6))==0
        warning(['IOP not properly set'])
    end
    
    IOP
    
    % For IPP, assume origin(+position) is at location row and column N/2+1
    % Compute TLC as IPP, here assuming any readout oversampling has been
    % removed.
    IPP = -PS_HW(2)*floor((reconSpace.matrixSize.x + 1)/2) * IOP(1:3) + ...
          -PS_HW(1)*floor((reconSpace.matrixSize.y + 1)/2) * IOP(4:6) + ...
           position(:,iaq)'
end


    
    
    
    



