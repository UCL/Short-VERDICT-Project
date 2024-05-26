function meas = dyn_summary(vdyn, mtimes)
% DYN_SUMMARY Summary of dynamic dual echo data
%   meas = dyn_summary(vdyn, mtimes)
%  vdynn [ny nx nz ndyn necho]
%  mtimes 2-element vector [aif_start  aif_end]
%
% meas is a structure with outputs
%  aifpeak1, aifpeak2  The area between the curve and the baseline over the
%  time points in mtimes (not scaled)
%
%  slope1, slope2  slope from end of aif to finish
%  uptake1, uptake2  difference between average at end and av at baseline
%  maxdip2       maximum dip in signal 
%
% 
% D.Atkinson@ucl.ac.uk


nlast = 10 ;  % number of points averaged at end for enhancement calc

base = vdyn(:,:,:,1:mtimes(1)-1,:) ;
baseav = sum(base,4)/(mtimes(1)-1) ;

last = vdyn(:,:,:,end-(nlast-1):end,:) ;
last  = sum(last,4)/nlast ;

signal = vdyn(:,:,:,[mtimes(1):mtimes(2)],:) ;
ntp = size(signal,4) ;

aifpeak = sum(signal-repmat(baseav,[1 1 1 ntp 1]),4) ;
meas.aifpeak1 = aifpeak(:,:,:,1,1) ;
meas.aifpeak2 = aifpeak(:,:,:,1,2) ;


% Slopes post AIF
slope = vdyn(:,:,:,end,:) - vdyn(:,:,:,mtimes(2)+1,:) / ...
    (size(vdyn,4)-(mtimes(2)+1)) ;

meas.slope1 = slope(:,:,:,1,1) ;
meas.slope2 = slope(:,:,:,1,2) ;

meas.uptake1 = last(:,:,:,1,1) -baseav(:,:,:,1,1) ;
meas.uptake2 = last(:,:,:,1,2) -baseav(:,:,:,1,2) ;

meas.maxdip2 = max(repmat(baseav(:,:,:,1,2),[1 1 1 size(vdyn,4) 1]) - vdyn(:,:,:,:,2) , [], 4);
