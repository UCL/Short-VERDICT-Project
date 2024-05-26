function [ datat, geomt ] = dinplanet( data, geom, transform, val )
%DINPLANET In plane transformations for DICOM data
%   The same transformation is applied to all dimension higher than 2 in
%   data. Except slice reversal is also possble for flip 3.
%
%  [ datat, geomt ] = dinplanet( data, geom, transform, val )
%
% data [nx ny ...]
% geom [1 nsl]
%
% geom pre-filled geom with basic DICOM info (note XData and any extra 
%   fields will be removed)
% 
%  transform  val
%  'rot', val must be 1 or -1
%  'flip' val must be dim: 1 (row), 2 (col) or 3
%  'circshift' val must be two element vector of row and col shifts
%  'Philips' val must be 3 
%  'remove_oversampling'  val must be three element vector of
%     recon_resolutions (will also remove oversampled slices if 3D).
%  'crop'  val [xlim ylim width height] (typically from an roi position
%               xlim, ylim refer to top left of cropped in units of original.
%  'zeropad'
%  'imresize'  val must be  [numrows numcols]
%
%

% Philips enumerations for converting from MP(S) to RC(V)
% From mggrcd.h
% DEFINE_GOAL_ENUM(MGG_RC_IN_PLANE_TRANSF_ENUM, "XEGRC02",
%   1  (MGG_RC_IN_PLANE_TRANSF_PLUS_A_PLUS_B)
%   2  (MGG_RC_IN_PLANE_TRANSF_PLUS_A_MIN_B)
%   3  (MGG_RC_IN_PLANE_TRANSF_MIN_A_PLUS_B)
%   4  (MGG_RC_IN_PLANE_TRANSF_MIN_A_MIN_B)
%   5  (MGG_RC_IN_PLANE_TRANSF_PLUS_B_PLUS_A)
%   6  (MGG_RC_IN_PLANE_TRANSF_PLUS_B_MIN_A)
%   7  (MGG_RC_IN_PLANE_TRANSF_MIN_B_PLUS_A)
%   8  (MGG_RC_IN_PLANE_TRANSF_MIN_B_MIN_A)
%
% Seems to mean:
%  [R C] = [M P] [A1  A2
%                 B1  B2 ]
%
% So MIN_A_MIN_B would mean  [ -1 0
%                               0 -1]
% i.e. row is -M, column is -P
% So with M currently the first index in reconstructed image, rot90 and
% fliplr should fix it.


nsl = length(geom(:)) ;

% Create blank geom to avoid risk of copying when should be modified
geomt = struct('IPP',num2cell(zeros([nsl 3]),2), ...
    'IOP',num2cell(zeros([nsl 6]),2), ...
    'Height',num2cell(zeros([nsl 1])), ...
    'Width',num2cell(zeros([nsl 1])), ...
    'PixelSpacing_HW',num2cell(zeros([nsl 2]),2), ...
    'SliceThickness',num2cell(zeros([nsl 1])), ...
    'MRAqType', num2cell(repmat('??',[nsl 1]),2), ...
    'source',{'Data source'}) ;

% 

warned2nd = false ;

for isl = 1:nsl
    if isfield(geom(isl),'source')
        geomt(isl).source = geom(isl).source ;
    end
    if isfield(geom,'MRAqType')
      geomt(isl).MRAqType = geom(isl).MRAqType ;
    end
    geomt(isl).SliceThickness = geom(isl).SliceThickness ;
    
    %check parallel
    if norm(geom(isl).IOP - geom(1).IOP) > 0.01
        warning(['Slices are not parallel'])
    end
    
   iop = geom(isl).IOP ;
   ipp = geom(isl).IPP ;
   sziop = size(iop) ; szipp = size(ipp) ;
   % make both row vectors for now
   iop = iop(:) ; ipp = ipp(:) ;
   
   PS_HW = geom(isl).PixelSpacing_HW ;
    
    switch transform
        case 'imresize'
            geomt(isl).Height = val(1) ;
            geomt(isl).Width =  val(2) ;
            geomt(isl).PixelSpacing_HW(1) = PS_HW(1)*geom(isl).Height / geomt(isl).Height ;  
            geomt(isl).PixelSpacing_HW(2) = PS_HW(2)*geom(isl).Width / geomt(isl).Width  ;
            
            geomt(isl).IOP = reshape(iop, sziop) ;
            
            newipp = ipp - 0.5*(PS_HW(2)*iop(1:3) + PS_HW(1)*iop(4:6)) + ...
                0.5*(geomt(isl).PixelSpacing_HW(2)*iop(1:3) + geomt(isl).PixelSpacing_HW(1)*iop(4:6)) ;
            geomt(isl).IPP = reshape(newipp, szipp) ;
            
            datat = imresize(data, val) ;
            
        case 'flip'
            geomt(isl).Height = geom(isl).Height ;
            geomt(isl).Width = geom(isl).Width ;
            geomt(isl).PixelSpacing_HW = PS_HW ;
            
            switch val
                case 2 % left right
                  iopt(1:3) = -iop(1:3) ; % flip row direction
                  iopt(4:6) = iop(4:6) ;
            
                   % new IPP is old top right pixel
                   ippt = ipp + ((geom(isl).Width-1) * PS_HW(2) * iop(1:3)) ;
                
                case 1 % up down
                    iopt(1:3) = iop(1:3) ; % flip column direction
                    iopt(4:6) = -iop(4:6) ;
            
                     % new IPP is old bottom left pixel
                     ippt = ipp + ((geom(isl).Height-1) * PS_HW(1) * iop(4:6)) ;
                case 3
                    % do nothing - handled later in bulk.
                    iopt = iop ;
                    ippt = ipp ;
                otherwise
                    error(['For flip, val must be 1 (up) or 2 (lr)'])
            end
            
            geomt(isl).IOP = reshape(iopt, sziop);
            geomt(isl).IPP = reshape(ippt, szipp) ;
            
            if isl == 1
              datat = flip(data, val) ; % only do this once
            end
            
        case 'rot'
            % swap width, height, pixelspacings
            geomt(isl).Height = geom(isl).Width ;
            geomt(isl).Width = geom(isl).Height ;
            geomt(isl).PixelSpacing_HW = PS_HW([2 1]) ;
            
            switch val
                case 1
                    % rot90 is counter clockwise
                    iopt(1:3) = iop(4:6) ;
                    iopt(4:6) = -iop(1:3) ;
                    
                    % new top left is old top right
                    ippt = ipp + ((geom(isl).Width-1) * PS_HW(2) * iop(1:3)) ;
                case -1
                    % rot90clock is  clockwise
                    iopt(1:3) = -iop(4:6) ;
                    iopt(4:6) = iop(1:3) ;
                    
                    % new top left is old bottom left
                    ippt = ipp + ((geom(isl).Height-1) * PS_HW(1) * iop(4:6)) ;
                otherwise
                    error(['For rot, val must be 1 (counter clock) or -1 (clock)'])
            end
            
            geomt(isl).IOP = reshape(iopt, sziop);
            geomt(isl).IPP = reshape(ippt, szipp) ;
            
            if isl == 1
              datat = rot90(data, val) ; % only do this once
            end
            
        
        case 'circshift'
            % preserves everything except ipp
            geomt(isl).Height = geom(isl).Height ;
            geomt(isl).Width = geom(isl).Width ;
            geomt(isl).PixelSpacing_HW = PS_HW ;
            geomt(isl).IOP = reshape(iop, sziop) ;
            
            ippt = ipp - (val(1)*iop(4:6)*PS_HW(1) + val(2)*iop(1:3)*PS_HW(2)) ;
            
            geomt(isl).IPP = reshape(ippt, szipp) ;
            
            if isl ==1  % Only shift once
                datat = circshift(data,val(1),1) ;
                datat = circshift(datat, val(2),2) ;
            end
            
        case 'crop'
            % val is [xlim ylim width height] from roi Position
            % xl, yl  refer to top left of cropped in units of original.
            % 

            yl = round(val(2)) ;
            yl = max(1, min(geom(isl).Height, yl)) ;
            
            yu = yl + round(val(4)) ;
            yu = max(1, min(geom(isl).Height, yu)) ;
            
            xl = round(val(1)) ;
            xl = max(1, min(geom(isl).Width, xl)) ;
            
            xu = xl + round(val(3)) ;
            xu = max(1, min(geom(isl).Width, xu)) ;
            
            geomt(isl).Height = yu-yl+1 ;
            geomt(isl).Width  = xu-xl+1 ;
            geomt(isl).PixelSpacing_HW = PS_HW ;
            geomt(isl).IOP = reshape(iop, sziop) ;
            
            ippshift = (yl-1)*PS_HW(1)*iop(4:6) + ...
                (xl-1)*PS_HW(2)*iop(1:3) ;
            
            ippt = ipp + ippshift ;
            
            geomt(isl).IPP = reshape(ippt, szipp) ;
            
            % Apply to all dimensions
            szdata = size(data) ;
            if isl ==1 
                datat = data(yl:yu,xl:xu,:) ;
                datat = reshape(datat,[geomt(isl).Height, geomt(isl).Width,  szdata(3:end)]) ;
            end
            
        case 'Philips'
            % These are the enumerated Philips in_plane_transformations to
            % go from MPS to RCV.
            % ! Zero-based in sin file, here 1-based !
            switch val
                case 1 % PLUS_A_PLUS_B
                    % R = M, C=P
                    [ datat, geomt ] = dinplanet( data, geom, 'rot', 1 ) ;
                    [ datat, geomt ] = dinplanet( datat, geomt, 'flip', 1 ) ;
                case 2 % PLUS_A_MIN_B
                    % R=M, C=-P
                    [ datat, geomt ] = dinplanet( data, geom, 'rot', 1 ) ;
                    
                case 3 % MIN_A_PLUS_B
                    % R=-M, C=P
                    [ datat, geomt ] = dinplanet( data, geom, 'rot', -1 ) ;
                    
                case 4 % MIN_A_MIN_B 
                    % R=-M, C=-P
                    [ datat, geomt ] = dinplanet( data, geom, 'rot', 1 ) ;
                    [ datat, geomt ] = dinplanet( datat, geomt, 'flip', 2 ) ;
                case 5 % PLUS_B_PLUS_A
                       %  [R C] = [M P] [0  1
                       %                 1  0 ]
                       %  R = P ,  C=M
                       datat = data ;
                       geomt = geom ;
                case 6 % PLUS_B_MIN_A
                       % R=P, C=-M
                    [ datat, geomt ] = dinplanet( data, geom, 'flip', 1 ) ;   
                case 7 % MIN_B_PLUS_A
                    % R=-P, C=M
                    [ datat, geomt ] = dinplanet( data, geom, 'flip', 2 ) ;
                case 8 % MIN_B_MIN_A
                    % R=-P , C=-M
                     [ datat, geomt ] = dinplanet( data, geom, 'flip', 2 ) ;
                     [ datat, geomt ] = dinplanet( datat, geomt, 'flip', 1 ) ;
                otherwise
                    error(['Philips in_plane_transformation enum not yet implemented: ',num2str(val)])
            end
            
        case 'remove_oversampling'
            % perform 2D in-plane chopping
            % through slice done later
            recon_res = val ;
            nd1 = size(data,1) ;
            nd2 = size(data,2) ;
            
            if nd1< recon_res(1) || nd2 < recon_res(2)
                error(['Cannot chop smaller image'])
            end
            
            % Issue warning if not equal numbers removed on each side
            if mod(nd1,2) ~= mod(recon_res(1),2)
                warning(['First dimension not chopped equally'])
            end
            
            if ( mod(nd2,2) ~= mod(recon_res(2),2) ) && ~warned2nd
                warning(['Second dimension not chopped equally'])
                warned2nd = true ;
            end
            
            % Align DCgps for full and chopped images.
            %! x here is first diemnsion
            DCgpx = DCgp(data,1) ; DCgpy = DCgp(data,2) ;
            DCgpx_out = DCgp([1:recon_res(1)],2) ;
            DCgpy_out = DCgp([1:recon_res(2)],2) ;
            
            % 1 2 3 4 5 6 7 8
            %   1 2 3 4 5 6
            
            % 1 2 3 4 5 6 7 8 9
            %   1 2 3 4 5 6
            
            geomt(isl).PixelSpacing_HW = PS_HW ;
            geomt(isl).IOP = reshape(iop, sziop) ;
            geomt(isl).Width = recon_res(2) ;
            geomt(isl).Height = recon_res(1) ;
            
            ippt = ipp + PS_HW(1)*iop(4:6)*(DCgpx-DCgpx_out) + ...
                PS_HW(2)*iop(1:3)*(DCgpy-DCgpy_out) ;
            
            geomt(isl).IPP = reshape(ippt, szipp) ;
            
            szdata = size(data) ;
            if isl ==1
                datat = data(DCgpx-DCgpx_out+1 : DCgpx-DCgpx_out+recon_res(1), ...
                    DCgpy-DCgpy_out+1 : DCgpy-DCgpy_out+recon_res(2) ,: ) ;
                datat = reshape(datat,[recon_res(1) recon_res(2) szdata(3:end)]) ;
            end
            
        case 'zeropad'
            % Image padding. Assume prior to rotations and flipping so DC
            % point is in the 'expected' array position and M and P are the
            % expected axes
            M_extra = val(1) ;
            P_extra = val(2) ;
            
            nM = geom(isl).Height ;
            nP = geom(isl).Width ;
            
            % Work out where to add the extra zeros
            if rem(nM,2) == 0
                if rem(M_extra,2) == 0
                    Mlow = M_extra/2;
                    Mhigh = M_extra / 2;
                else
                    Mlow = (M_extra-1)/2 ;
                    Mhigh = (M_extra+1)/2 ;
                end
            else
                if rem(M_extra,2) == 0
                    Mlow = M_extra/2;
                    Mhigh = M_extra / 2;
                else
                    Mlow = (M_extra+1)/2 ;
                    Mhigh = (M_extra-1)/2 ;
                end
            end
            
            if rem(nP,2) == 0
                if rem(P_extra,2) == 0
                    Plow = P_extra/2;
                    Phigh = P_extra / 2;
                else
                    Plow = (P_extra-1)/2 ;
                    Phigh = (P_extra+1)/2 ;
                end
            else
                if rem(P_extra,2) == 0
                    Plow = P_extra/2;
                    Phigh = P_extra / 2;
                else
                    Plow = (P_extra+1)/2 ;
                    Phigh = (P_extra-1)/2 ;
                end
            end
            
            ippt = ipp - iop(4:6)*Mlow*PS_HW(1) - iop(1:3)*Plow*PS_HW(2);
            
            geomt(isl).Height = geom(isl).Height + M_extra ;
            geomt(isl).Width = geom(isl).Width + P_extra ;
            geomt(isl).PixelSpacing_HW = PS_HW ;
            geomt(isl).IOP = reshape(iop, sziop) ;
            geomt(isl).IPP = reshape(ippt, szipp) ;
            
            szdata = size(data) ;
            if isl ==1
                disp(['Zero pad by ',num2str(M_extra),' in M, and ',...
                    num2str(P_extra),' in P.'])
                datat = zeros([szdata(1)+M_extra, szdata(2)+P_extra, prod(szdata(3:end))]) ;
                
                datat(1+Mlow: end-Mhigh, 1+Plow: end-Phigh,:) = data(:,:,:) ;
                
                datat = reshape(datat,[szdata(1)+M_extra, szdata(2)+P_extra, szdata(3:end)]) ;
            end
            
        otherwise
            error(['Transform not implemented; ',transform])
    end
    
end  % end isl loop

switch transform
    case 'remove_oversampling'
        % 3D oversampling removal
        recon_res = val ;
        nd3 = size(data,3) ;
        if nd3 > 1
            if nd3< recon_res(3)
                error(['Cannot chop smaller image in S '])
            end
            
            % Issue warning if not equal numbers removed on each side
            if mod(nd3,2) ~= mod(recon_res(3),2)
                warning(['3rd dimension not chopped equally'])
            end
            
            DCgpz = DCgp(data,3) ;
            DCgpz_out = DCgp([1:recon_res(3)],2) ;
            
            szdata = size(data) ;
            
            datat = datat( : , :, DCgpz-DCgpz_out+1 : DCgpz-DCgpz_out+recon_res(3) ,: ) ;
            datat = reshape(datat,[recon_res(1) recon_res(2) recon_res(3) szdata(4:end)]) ;
            
            % prune out the unwanted slices
            geomt(1:DCgpz-DCgpz_out) = [] ;
            geomt(recon_res(3)+1 : end) = [] ;
        end
    case 'flip'
        if val == 3
            % data is already flipped above, just do geom
            geomt = geomt(end:-1:1) ;
        end     
end

end

