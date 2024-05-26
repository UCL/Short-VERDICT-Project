function csens = sense(sdata, sinfo, csdata, csinfo, base_geom)
% SENSE Calculate coil sensitivities
%
% csens = sense(sdata, sinfo, csdata, csinfo)
% csens = sense(sdata, sinfo, csdata, csinfo, base_geom)
%
% Does not yet do virtual coils. See Buehrer et al:
% MRM 57:1131 - 1139 (2007)
%

% To Do:
%  Correctly sort the coil receiver_nrs, channel_names etc.
%   compare REconFrame and this
%    do fitting on these, improve mask
%
% In example,
% Coil Survey 2 stacks
% 34 channel names (1st 2 are QBC)
% 34 receiver numbers (first two 27 30)
% Stack 1 measured_channels 0 1
% Stack 2 measured_channels 2 - 33
% dc_fixed_arr has 32 entries! implies no entry for QBC
%
% SENSE ref scan  2 stacks
% 34 channel_names (last two QBC)
% 34 Receivers 0 - 31, then 27 30
% Stack 1 measured_channels 0 - 31
% Stack 2 measured_channels 0 - 31
%
% DWI
% 33 channel names
% 1st anterior is absent in channel_names, last two QBC
% 33 receiver numbers 1 - 31, 27 30
% 31 measured channels 0 - 30
% dc_fixed_arr has 32 entries, but first is zero. 

% Hypothesis is that data in raw profile has length measured_channels 
% for that stack. 
%   measured_channels(idxraw) -> idxchannel
%   receiver_numbers(idxchannel) -> index for array comparable between main
%      scan and SENSE?
%  
%   channel_names(idxchannel) -> coil element name 
%
% Restore % dc_fixed = info_loc.sin.dc_fixed_arr(receiverNr); in step6
% in step1, remove at line 754 the bit that removes 0 dc_fixed_arr
% nCoils is set to nr_measured_channels - but this is not number receivers.
% Needs a new branch and cool head!
%
% In The Philips code, nCoils is nr_measured_channels and there is one per
% stack.
% measured_channels_of_stacks is {stack}(idxraw)
%



[isdata, sgeom ] = recon_process(sdata, sinfo,'loc',1,'kbeta',5);
[csvolqbc, csgeom ] = recon_process(csdata, csinfo,'loc',1,'coil',[1 2], ...
    'kbeta',5);

if nargin < 5
    base_geom = sgeom ;
end

% Quadrature combination
qcsvolqbc = csvolqbc(:,:,:,1) -1i*csvolqbc(:,:,:,2) ;

body_ref = qcsvolqbc ;

csatbase = dreslice(body_ref,csgeom, base_geom) ;
satbase  = dreslice(isdata, sgeom, base_geom) ;

eshow(csatbase,'Name','Coil Survey ref')
eshow(sqrt(sum(satbase.*conj(satbase),4)),'Name','SOS')

csmax = max(abs(csatbase(:))) ;
loc  = find(abs(csatbase) < csmax/100) ;
csatbase(loc) = NaN ;
csens = satbase ./ csatbase ; % implicit expansion in coil dim
csens(isnan(csens)) = 0 ;

end

function idata = spec_orig(idata, info)
if ~isempty(info.sin.spectrum_origins)
    % Do not update geom as this correction should have been applied
    % earlier in recon (but is not).
    spectrum_origins = info.sin.spectrum_origins{1,1}(1:3) ;
    idata = circshift(idata, spectrum_origins(1),1) ;
    idata = circshift(idata, spectrum_origins(2),2) ;
    idata = circshift(idata, spectrum_origins(3),3) ;
end
end


