function [vol, matp_rs] = dresamp(vol_in, matp_in, matp_out, varargin)
% DRESAMP EDITED DRESLICE USE DRESLICE!!! Resample data, including vector fields. Investigational use only.
%
%  [volrs] = dresamp(vol_in, geom_in, geom_out)
%  [volrs, geom_rs] = dresamp(vol_in, geom_in, geom_out, param, value,...)
%
% vol_in   Input volume whose intesity values will be interpolated. 
%          Must have third dimension of slice. May have 4th and 5th
%          dimensions for vector or tensor field. [ny nx nz D1 D2]
% geom_in  Structure containing the geometry parameters of vol_in (likely
%          this comes from d2mat when data is read in from files). 
% geom_out Structure containing the geometry parameters of the output 
%          volume volrs. Likely to come from the reading in with d2mat of
%          another data set.
% volrs    The output volume, i.e. vol_in resliced to the geometry
%          specified in geom_out. [... D1 D2]
% geom_rs  If you want the plane(s) and FOV of geom_out, but the resolution
%          of the input volume, then specify 'PixelSpacing' 'input' 
%          (see below). Typically when vol_in is a high res volume and you 
%          want to keep the resolution, but interpolate to the plane of, 
%          say, a diffusion slice. In this case, geom_rs is
%          similar to geom_out, but has updated information for the pixel
%          sizes.
%          The SliceThickness in the geoms supplied by the user are ignored
%           - they are set by looking at the IPP, IOP and distance between
%           slices in geom_check
%
% The geom structures can be replaced by a structure that contains
% a geom field (e.g. from d2mat). geom need to contain a MRAqType  
%
% In plane interpolation uses a cubic kernel defined over 200 points 
% in the range 0 to 2. The kernel is adjusted to prevent aliasing. 
% The through-plane kernel is also cubic for 3D data. For 2D it is linear 
% unless there is only one slice, in which case it is nearest neighbour.
%
% Parameter value pairs:
% 'PixelSpacing', 'input'   Sets the output pixel spacing to be the same as
% the input.
% 'quiet', true | { false }
%
% Example usage:
%   dinfo_in = datparse ;
%   [vol_in, matp_in] = d2mat(dinfo_in,{'slice'},'op','dv') ;
%   
%   dinfo_comp = datparse ;
%   [vol_comp, matp_comp] = d2mat(dinfo_comp,{'slice'},'op','dv') ;
%
%   [vol] = dreslice(vol_in, matp_in, matp_comp) ;
%
% Example.
% To resample a high res T2W to correspond to a lower res diffusion image,
% the input is the T2W image and geom, geom_out is the diffusion. To
% preserve the in-plane resolution, use parameter/value pair,
% 'PixelSpacing','input'
%   [vol, matp_rs] = dreslice(vol_in, matp_in, matp_out, ...
%                                'PixelSpacing','input') ;
% Note that to keep the FOV of the input, the new PixelSpacing may not 
% be exactly the same as that requested. geom_out will be updated.
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
% David Atkinson D.Atkinson@ucl.ac.uk
%
% See also DATPARSE D2MAT DGEOMEXTRACT GEOM_CHECK
%

ps_input = 0 ; % Pixel Spacing to use input
quiet = false ;

for ipv = 1:2:length(varargin)
    switch varargin{ipv}
        case 'PixelSpacing'
            if nargout < 2
                warning(['geom_out may be modified but only 1 output requested'])
            end
            
            switch varargin{ipv+1}
                case {'input', 'Input'}
                    ps_input = 1 ;
                otherwise
                    disp(['Pixel Spacing will be as in geom_out'])
            end
            
        case 'quiet'
            quiet = varargin{ipv+1} ;
        otherwise
            warning(['Unrecognised parameter: ',varargin{ipv}])
    end
end

if isfield(matp_in,'geom')
    geom_in = matp_in.geom ;
else
    geom_in = matp_in ;
end
if isfield(matp_out,'geom')
   geom_out = matp_out.geom ;
else
    geom_out = matp_out ;
end


if size(vol_in,3) ~= length(geom_in)
    warning(['vol_in and geom should have corresponding number of slices.'])
end

% Check parallel and equal spaced slices, get slice separation.
% The 'f' and 'q' refer to 'fixed' and 'queried'.
% The fixed input is queried at the output locations.
% rowf is along a row, colf is down a column
[slsep_in,  rowf, colf, slcf] = geom_check(geom_in) ; % row is iop(1:3)
[slsep_out, rowq, colq, slcq] = geom_check(geom_out) ;

nsl_out = length(geom_out) ;

if ps_input ~= 0 % use pixel spacing of input geom
    % modify PSq, geom_out PixelSpacing, Height and Width
    PS_HW_old = geom_out(1).PixelSpacing_HW ;
    FOVf_HW = [ geom_out(1).PixelSpacing_HW(1)*double(geom_out(1).Height) ...
        geom_out(1).PixelSpacing_HW(2)*double(geom_out(1).Width) ] ;
    
    nHeight = round(FOVf_HW(1)/geom_in(1).PixelSpacing_HW(1)) ;
    nWidth  = round(FOVf_HW(2)/geom_in(1).PixelSpacing_HW(2)) ;
    
    % set Height, Width, PixelSpacing geom_out (all slices)
    geom_out(1).PixelSpacing_HW(1) =  FOVf_HW(1)/nHeight ;
    geom_out(1).PixelSpacing_HW(2) =  FOVf_HW(2)/nWidth ;
    geom_out(1).Height = nHeight ;
    geom_out(1).Width = nWidth ;
    
    PS_HW_new = geom_out(1).PixelSpacing_HW ;
    
    
    for ig = 1:nsl_out
        geom_out(ig).PixelSpacing_HW = PS_HW_new ;
        geom_out(ig).Width = geom_out(1).Width ;
        geom_out(ig).Height = geom_out(1).Height ;
        
        %Update IPP, XData and YData
        % compute top left corner (not pixel centre) for these calculations
        ipp = geom_out(ig).IPP;
        iop = geom_out(ig).IOP ;
        tlc = ipp - 0.5*(PS_HW_old(2)*iop(1:3) + PS_HW_old(1)*iop(4:6)) ;
        
        IPPnew = tlc + 0.5*(PS_HW_new(2)*iop(1:3) + PS_HW_new(1)*iop(4:6)) ;
        
        %Computation of XData etc copied from dgeomextract
        plane = set_plane('pvv',ipp,iop(1:3),iop(4:6)) ;
        d = point_plane([0 0 0],plane) ; %Patient coord origin to plane distance
        oip = d*plane.normal ; %closest point in plane of origin
        oip2ipp = ipp - oip ;
        XData(1) = dot(oip2ipp,iop(1:3)) ;
        YData(1) = dot(oip2ipp,iop(4:6)) ;
        XData(2) = XData(1) + PS_HW_new(2)*(nWidth-1) ;
        YData(2) = YData(1) + PS_HW_new(1)*(nHeight-1) ;
        
        geom_out(ig).IPP = IPPnew ;
        geom_out(ig).XData = XData ;
        geom_out(ig).YData = YData ;
        
        [R2D,D] = geom2sro(geom_out(ig)) ;
        geom_out(ig).R2D = R2D;
        geom_out(ig).D = D ;
    end
    
    
end

sz_vol_in = size(vol_in) ;

% define PixelSpacing F in "fixed" input data and Q queried in rcs order
% so it's width height slice 
PSf = [geom_in(1).PixelSpacing_HW(2)  geom_in(1).PixelSpacing_HW(1)  slsep_in] ;
PSq = [geom_out(1).PixelSpacing_HW(2) geom_out(1).PixelSpacing_HW(1) slsep_out] ;

vol = zeros([geom_out(1).Height geom_out(1).Width nsl_out sz_vol_in(4:end)]) ;

% Need to sort out dimensions for resampler to cope with non-orthogonal resampling
% griddedInterpolant looks interesting but it is not clear that the kernel
% will follow the correct shape for aliasing.
% Note that the PixelSpacings used for the positions of the output are not
% affected by the code below here, this all to do with the kernel and its
% scaling.

% 
rowdots = abs([dot(rowq,rowf) dot(colq,rowf) dot(slcq,rowf)]) ;
coldots = abs([dot(rowq,colf) dot(colq,colf) dot(slcq,colf)]) ;
slcdots = abs([dot(rowq,slcf) dot(colq,slcf) dot(slcq,slcf)]) ;

tcoldots = coldots ;
tslcdots = slcdots ;

% Selects PixelSpacing direction closest to output? 
[Rd,iR] = max(rowdots) ;
tcoldots(iR) = 0 ; tslcdots(iR) = 0 ;  % designed to resolve 45degree cases
[Cd,iC] = max(tcoldots) ;
tslcdots(iC) = 0 ;
[Sd,iS] = max(tslcdots) ;

u = unique([iR iC iS]) ;
if length(u) ~= 3
    warning(['should be 3 unique direction'])
end

% Forthcoming sets the kernel size. This is a separable kernel used for
% interpolation in the tformarray command. As coded, tformarray operates in
% 3D image space where the distance between voxels is 1 in every dimension.
% To avoid aliasing, the kernel must scale.
%
% Old code: PS_out = [PSq(iR)*Rd  PSq(iC)*Cd  PSq(iS)*Sd] ;
% But the "rowf" etc above is ALONG a row, i.e. should be the 2nd 
% dimension, hence the swap below.
% The division is because when the slice is angulated the kernel needs to
% be bigger (for angulated slices Cd, Rd and or Sd will be a little less
% than 1).

% PS_out = [ PSq(iC)/Cd  PSq(iR)/Rd  PSq(iS)*Sd] ; % Appears incorrect in
% last element?

% PS_out = [ PSq(iC)/Cd  PSq(iR)/Rd  PSq(iS)/Sd] ;
PS_out = [ PSq(iR)/Rd  PSq(iC)/Cd  PSq(iS)/Sd] ;

kv = cubic(linspace(0,2,200)) ; % cubic half kernel from 0 to 2 for makeresampler

scale = PS_out ./ PSf ; %!!!!! Psf and PS_out in different order??!!

rscale = max([1 1 1],scale) ;
disp(['rscale: ',num2str(rscale)])

% rscale = [rscale(2) rscale(1) rscale(3)]

% makeresampler takes as inputs {half_width, positive_half}

% Possible through-slice scenarios:
%   True 3D input data: should use cubic or similar
%   2D MS or M2D with no gaps: linear
%   Slices with gaps: linear
%   Single slice: nearest neighbour with width of slice thickness

if size(vol_in,3) == 1 % single slice so nearest neighbour (box)
    khw = 1 / 2 ;
    kph = [1 1] ;
else
    switch geom_in(1).MRAqType
        case '3D' % cubic
            khw = 2*rscale(3) ;
            kph = kv/rscale(3) ;
        case '2D' % rect (linear)
            khw = rscale(3) ;     % This is slsep_in*rscale(3) in real units.
            kph = linspace(1,0,100) ;   
        otherwise % Non MR, hence use linear
            khw = rscale(3) ;     % This is slsep_in*rscale(3) in real units.
            kph = linspace(1,0,100) ;   
    end
end
    
R = makeresampler({{2*rscale(1) kv/rscale(1)}, ...
                   {2*rscale(2) kv/rscale(2)} , ...
                   {khw         kph        }} ,'fill');
               
T111 = [1 0 0 0 ; 
            0 1 0 0 ; 
            0 0 1 0 ; 
            1 1 1 1 ]; % translation of 1,1,1
        
Tscale = [1/PSf(1) 0 0 0 ; 
              0 1/PSf(2) 0 0 ; 
              0 0 1/PSf(3) 0 ; 
              0 0 0 1] ;
    
Trim = [rowf(1) colf(1) slcf(1) 0 ;
        rowf(2) colf(2) slcf(2) 0 ;
        rowf(3) colf(3) slcf(3) 0 ;
         0      0         0     1 ];
      
origin_pp = geom_in(1).IPP ;

    
Torig = [1 0 0 0 ; 
         0 1 0 0 ; 
         0 0 1 0 ; 
      -origin_pp(:)' 1] ;

% Ordering of rotation and scaling. Appears to work for rot then scale.
% Explanation:
%   MT converts a 3D LPH coordinate to an image coordinate system that is
%   aligned with the input image. It will be used to convert the 3D
%   coordinates of the output (where we want pixels) to the image
%   coordinate system. tformarray will do the interpolation.
%   The pixel scaling is essentially expressed in the image system, i.e a
%   distance along image x and y so it is applied after the rotation.
%
% It should be that if the input data 3D coordinates are post multiplied by
% MT, we just get integer image coordinates

% The interpolation kernel in-plane:
%  We have data at regular points in the input image. We want to
%  interpolate at other points which are regular, but the grid may not be
%  parallel to the input grid. We need to avoid alaising artefacts if the
%  new points are more than the separation of the old.
%    
%

MT = Torig *  Trim * Tscale * T111 ;   

ddisplay(vol_in, geom_in, MT, geom_out, PSq, rowq, colq)

hw = waitbar(0,['dresamp: Processing ',num2str(nsl_out),' slices.']) ;

for isl = 1:nsl_out
   % loop over each output slice
   waitbar(isl/nsl_out,hw) ;
   
   ippsl = geom_out(isl).IPP ;
   
   tmapb = zeros([geom_out(isl).Height geom_out(isl).Width 3]) ;
   
   for ix = 1:double(geom_out(isl).Width)
       for iy = 1:double(geom_out(isl).Height)
           p3D = ippsl + PSq(1)*rowq*(ix-1) + PSq(2)*colq*(iy-1) ; %3D world
           
           % MT = Torig *  Tscale *Trim *  T111 ; 
           pim = [p3D(:)' 1] * MT ; % 3D image space
           tmapb(iy,ix,:) = [pim(2) pim(1) pim(3) ] ;
       end
   end
   % B = tformarray(A, T, R, TDIMS_A, TDIMS_B, TSIZE_B, TMAP_B, F) 
   % tformarray using tmapb to specify output locations 
   % F is fill value (0).
   % each of the entries in tmapb described by the first two dimensions, is
   % a 3D coordinate in image space 
   B = tformarray(vol_in, [], R, [1 2 3], [1 2], [], tmapb, 0) ;
   
   vol(:,:,isl,:,:) = B ;
   
end
close(hw)
if nargout > 1
    if isfield(matp_out,'geom')
      matp_rs = matp_out ;
    end
    matp_rs.geom = geom_out ;
    if ps_input ~= 0 % PixelSpacing is in but slice sep is out
        matp_rs.vdims = matp_in.vdims ;
        matp_rs.vdims(3) = matp_out.vdims(3) ;
    end


end

if ~quiet && exist('eshow','file') && length(size(vol))<4
  eshow(vol,'vdim',PSq,'Name',['(dresamp) resampled data'])
end
end

%---------------------------------
function f = cubic(x)
% From MATLAB imresize function:
% See Keys, "Cubic Convolution Interpolation for Digital Image
% Processing," IEEE Transactions on Acoustics, Speech, and Signal
% Processing, Vol. ASSP-29, No. 6, December 1981, p. 1155.

absx = abs(x);
absx2 = absx.^2;
absx3 = absx.^3;

f = (1.5*absx3 - 2.5*absx2 + 1) .* (absx <= 1) + ...
                (-0.5*absx3 + 2.5*absx2 - 4*absx + 2) .* ...
                ((1 < absx) & (absx <= 2));
end
    

function ddisplay(vol_in, geom_in, MT, geom_out, PSq, rowq, colq)
nsl = size(vol_in,3) ;
isl = ceil(nsl/2) ;
img = abs(vol_in(:,:,isl,1)) ;

nsl_out = length(geom_out) ;
isl_out = ceil(nsl_out/2) ;


figure('Name', ['(dresamp) Input sl: ',num2str(isl),', output: ',num2str(isl_out)]);

    nrow = size(vol_in,1) ;
    ncol = size(vol_in,2) ;
    ps = geom_in(isl).PixelSpacing_HW ;
    iop = geom_in(isl).IOP ;
    ipp = geom_in(isl).IPP ;
    
    
    % form matrices X,Y and Z to contain image
    % vertex coordinates
    X = zeros(nrow+1, ncol+1) ;
    Y = zeros(nrow+1, ncol+1) ;
    Z = zeros(nrow+1, ncol+1) ;
    
    % 3D coord of top left vertex.
    %
    %  tlv
    %  +-----+-----+-----+
    %  |     |     |     |
    %  |  x  |     |     |
    %  |     |     |     |
    %  +-----+-----+-----+
    %  |     |     |     |
    %  |     |     |     |
    %
    % x is centre of top left coord, this is
    % ipp in the patient coordinate system.
    % To compute tlv (top left vertex), go
    % back half a pixel and up half a pixel.
    % Scale for the pixel dimensions
    %
    % Take care over row/column spacing, directions etc
    tlv = ipp' + -0.5*(iop(1:3)*ps(2) + iop(4:6)*ps(1))  ;
    
    % loop over each pixel
    for ir = 1:nrow+1
        for ic = 1:ncol+1
            % v ertex coordinate is just the coord of the
            % top left vertex plus a vector down the rows
            % plus a vector along the columns.
            coordv = tlv + (ir-1)*iop(4:6)*ps(1) + ...
                (ic-1)*iop(1:3)*ps(2) ;
            
            X(ir,ic) = coordv(1) ; % separate the x component into the matrix X,
            Y(ir,ic) = coordv(2) ; % the y component into Y etc
            Z(ir,ic) = coordv(3) ;
        end
    end
    
    % Surf plot
    cm = colormap ;
    C = 1+floor( double(img)/max(double(img(:)))*(size(cm,1)-1) ) ;
    surf(X,Y,Z,C,'EdgeColor','None')
    drawnow
    hold on
    axis equal
    axis vis3d
    grid on
    view(21,-50)
    xlabel('x left') ; ylabel('y posterior') ; zlabel('z head')
    colormap gray
  
    % Bounding box for output in LPH
    
    ippsl = geom_out(isl_out).IPP ;
    %TL
    W = double(geom_out(isl_out).Width) ;
    H = double(geom_out(isl_out).Height) ;
    
    % Not sure about corners here. PSq is in WH order
    tl = ippsl + PSq(1)*rowq*(-0.5) + PSq(2)*colq*(-0.5) ;
    tr = ippsl + PSq(1)*rowq*(W-0.5) + PSq(2)*colq*(-0.5) ;
    bl = ippsl + PSq(1)*rowq*(-0.5) + PSq(2)*colq*(H-0.5) ;
    br = ippsl + PSq(1)*rowq*(W-0.5) + PSq(2)*colq*(H-0.5) ;
   
    % pim = [ tl(:)' 1] * MT ;
    pim = tl ;
    PL(1) = pim(1) ; PP(1) = pim(2) ; PH(1) = pim(3) ;
    
    %pim = [ tr(:)' 1] * MT ;
    pim = tr ;
    PL(2) = pim(1) ; PP(2) = pim(2) ; PH(2) = pim(3) ;
    
    %pim = [ bl(:)' 1] * MT ;
    pim = br ;
    PL(3) = pim(1) ; PP(3) = pim(2) ; PH(3) = pim(3) ;
    
    %pim = [ br(:)' 1] * MT ;
    pim = bl ;
    PL(4) = pim(1) ; PP(4) = pim(2) ; PH(4) = pim(3) ;
    
    PL(5) = PL(1) ; PP(5) = PP(1) ; PH(5) = PH(1) ;
    
    plot3(PL,PP,PH)
    plot3(ippsl(1), ippsl(2), ippsl(3),'b*')
    plot3(ipp(1), ipp(2), ipp(3), 'ro') 
    
    disp(['img is input, b* is IPP of output slice, ro is input'])
    disp(['ipp input: ',num2str(ipp(:)')])
    disp(['ipp output: ', num2str(ippsl(:)')])
    
end
       