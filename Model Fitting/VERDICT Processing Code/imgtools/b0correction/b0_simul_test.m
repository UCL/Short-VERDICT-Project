function b0_simul_test
% B0 Simulation Test
%
% Sequences are described in a seq structure. This contains Gradient
% structures that are assumed to be rectangular gradients. They are
% described by their start and end times and strength. 
% The FE read gradient and timings are assumed so that the signal is
% instantaneously sampled at the start of the gradient and then at every
% dwell/blip time.
%
%
% D.Atkinson@ucl.ac.uk
%

% Function that returns B0 field given spatial position, map model, field
% strength.
% Input: spatial position(s) in LPH coordinates
% Output: field in Tesla or Hertz?

% Consider spatial oversampling to prevent discretisation errors.

% Implement as EH and E



hw.B_T = 3 ; % main field 3T
hw.gamma_HzpT = 42.576e6 ;
hw.fat_ppm = 3.5 ;

seq.model= 'EPI';
seq.npe = 16 ;
seq.nfe = 16 ;
seq.dwell_us = 10 ; 
seq.blip_us = 10 ;
seq.RF2fe_rew_ms = 1 ;
seq.RF2pe_rew_ms = 1 ;
seq.FOVPE_mm = 220 ;
seq.FOVFE_mm = 220 ;
seq.peorder = 'up' ; % PE rewind has same area for up and down so 
                       % DC is at different ks-ace line

% FOR FUTURE USE
seq.offcentre_fe = 0 ;
seq.offcentre_pe = 0 ;


seq = buildseq(seq, hw) ;
plot_seq(seq) 
 
% ts = linspace(0,18e4,1000) ;
% for its = 1:length(ts) 
%     phit(its) = t2phi(ts(its), 10, 10, seq, hw) ; 
% end
% figure('Name','phase')
% plot(ts,phit)
% grid on

B0.B0_map_Hz  = zeros([seq.npe seq.nfe]) ;
B0.B0_map_Hz = 10*peaks(16) ; 

% test by simulating an acquisition
ksp = acq_seq(seq, B0, hw) ;

BWFEpPix_Hz = 1/(seq.Tacq_us * 1e-6 ) 


end

function ksp = acq_seq(seq, B0, hw)
% ACQ_SEQ Simulate acquiring sequence
% ksp = acq_seq(seq, hw)

npe = seq.npe ; nfe = seq.nfe ;
img = zeros([npe nfe]) ;
%img(20:100,40:90) = 3 ;
%img(25:95,50) = 6;
img(2:13,4:10)= 4 ;
img(4,4:10)  = 5 ;
img(7,4:10) = 5 ;
img(3,3) = 6 ;

B0_map_Hz = B0.B0_map_Hz ;

ksp = zeros([npe nfe]) ;

DCy = ceil((npe+1)/2) ;
DCx = ceil((nfe+1)/2) ;
pix_x = seq.FOVFE_mm/seq.nfe ;
pix_y = seq.FOVPE_mm/seq.npe ;

pix_kx = 1/seq.FOVFE_mm ;
pix_ky = 1/seq.FOVPE_mm ;

hwait = waitbar(0,'DFT');
for ikpe =1:npe
    waitbar(ikpe/npe, hwait)
    for ikfe = 1:nfe
        kpe = pix_ky * (ikpe-DCy) ;
        kfe = pix_kx * (ikfe-DCx) ;
        S = 0 ;
        
        currt = seq.k2t(ikpe, ikfe) ;
        
        for iy = 1:npe
            for ix = 1:nfe
                yc = pix_x * (iy-DCy) ;
                xc = pix_y * (ix-DCx) ;
                % Example DFT but without using gradients etc
                % S = S + img(iy,ix)*exp(-i*2*pi*(kpe*yc + kfe*xc) ) ; 
                
                [phi_seq, phife, phipe] = t2phi_seq(currt, xc, yc, seq, hw) ;
%                 disp(['ikpe ',num2str(ikpe),' ikfe ',num2str(ikfe),' xc ',num2str(xc), ...
%                     ' yc ',num2str(yc),' FE ',num2str(phife/xc),' ',num2str(2*pi*kfe), '  PE ',num2str(phipe/yc),' ',num2str(2*pi*kpe)])
%                 
                phi_B0 = t2phi_B0(B0_map_Hz, currt, xc, yc, seq);
                
                S = S + img(iy,ix)*exp(-1i*(phi_B0+phi_seq) ) ; 
            end
        end
        ksp(ikpe,ikfe) = S ;
    end
end
close(hwait)

eshow(k2i(ksp))

end

function phi = t2phi_B0(B0_map_Hz, currt, xc, yc, seq)
% T2PHI_B0
%
% D.Atkinson@ucl.ac.uk

XCV = seq.FOVFE_mm/seq.nfe * ([1:seq.nfe]-ceil((seq.nfe+1)/2)) ;
YCV = seq.FOVPE_mm/seq.npe * ([1:seq.npe]-ceil((seq.npe+1)/2)) ;

offset_Hz = interp2(XCV,YCV,B0_map_Hz,xc,yc) ;

phi = 2*pi*offset_Hz*currt*1e-6 ;

end

function [phi, phife, phipe] = t2phi_seq(currt, xc, yc, seq, hw)
% T2PHI_SEQ  Time in sequence to phase
%  [phi, phife, phipe] = t2phi_seq(currt, xc, yc, seq, hw)
% 
% phi = 2 pi f t = 2 pi gamma G x t

[gobj_pe, glastpe] = getgobj(seq.Gpe, currt) ;
[gobj_fe, glastfe] = getgobj(seq.Gfe, currt) ;

if ~isempty(gobj_fe)
    Afe = seq.Gfe(gobj_fe).Ast + (currt-seq.Gfe(gobj_fe).tst)*seq.Gfe(gobj_fe).str ;
else
    % Time is outside a FE gradient so area will be same at start of next
    % gradient
    % glastfe can empty at the start
    if isempty(glastfe)
        Afe = 0 ;
    else
        Afe = seq.Gfe(glastfe).Ast + seq.Gfe(glastfe).A ; 
    end
end

if ~isempty(gobj_pe)
    Ape = seq.Gpe(gobj_pe).Ast + (currt-seq.Gpe(gobj_pe).tst)*seq.Gpe(gobj_pe).str ;
else
    if isempty(glastpe)
        Ape = 0 ;
    else
        Ape = seq.Gpe(glastpe).Ast + seq.Gpe(glastpe).A ; 
    end
end

% xc, yc in mm
% hw.gamma_HzpT
% A in us.mT/m

phife = 2 * pi  * hw.gamma_HzpT  * Afe * xc * 1e-12;
phipe = 2 * pi  * hw.gamma_HzpT  * Ape * yc * 1e-12;

phi = phife + phipe ;

end

function seq = buildseq(seq, hw)
% BUILDSEQ Set gradient timings
% seq = buildseq(seq, hw)
%
% builds up the gradient objects given initial sequence parameters, sets
% their tst tfn and str fields
%
% David Atkinson  D.Atkinson@ucl.ac.uk
%

switch seq.model
    case 'EPI'
        
        Tacq_us = seq.dwell_us * (seq.nfe - 1) ; % w
        
        % Readout gradient strength
        G_fe_mTpm = 1/(hw.gamma_HzpT * seq.FOVFE_mm * seq.dwell_us ) *1e12; 

        % Phase encode gradient step
        % Gradient on for one blip moves us one k-space point in PE.
        switch seq.peorder
            case 'up'
                gsign = -1 ;
            case 'down'
                gsign = 1 ;
            otherwise
                error(['Unknown seq.peorder: ',seq.peorder])
        end
        G_pe_step_mTpm = ...
            gsign/(hw.gamma_HzpT * seq.FOVPE_mm * seq.blip_us ) * 1e12 ; 

        ngfe = seq.nfe + 1;
        ngpe = seq.npe + 1 ;
        
        % Assume rewind here is at same gradient strength as EPI. Its area
        % will be just over half of the readout on the assumption that
        % the first RO sample is at the start of the RO gradient, i.e. that
        % the rewind takes you to -65 in k-space for a 128 nfe.
        
        Gfe(1).tst = seq.RF2fe_rew_ms * 1000 ;
        Gfe(1).tfn = Gfe(1).tst + (ceil((seq.nfe-1)/2))*seq.dwell_us ; % was +1
        
        Gfe(1).str = -G_fe_mTpm ;
        
        Gpe(1).tst = Gfe(1).tst ;  % START of rewind blip to coincide with FE rewind
        Gpe(1).tfn = Gpe(1).tst + seq.blip_us ;
        
        Gpe(1).str = G_pe_step_mTpm * ceil((seq.npe-1)/2) ; % 
        
        Gpe(1).Ast = 0 ; Gpe(1).A = seq.blip_us * Gpe(1).str;
        Gfe(1).Ast = 0 ; Gfe(1).A = (Gfe(1).tfn - Gfe(1).tst) * Gfe(1).str;
        
        nfe = seq.nfe ; npe = seq.npe ;
        
        kvoddt =  [0:1:(nfe-1)]*seq.dwell_us ;
        k2t = zeros([npe nfe]) ;
        
        for ig = 2:ngfe ;
            
            if gsign == -1
                peline = ig-1;
            else
                peline = seq.npe - ig+2 ;
            end
            
            Gfe(ig).tst = Gfe(ig-1).tfn + seq.blip_us ;
            Gfe(ig).tfn = Gfe(ig).tst + Tacq_us ;
            
            if rem((ig-1),2)==1
                Gfe(ig).str = G_fe_mTpm ;
                k2t(peline, 1:nfe) = Gfe(ig).tst + kvoddt ;
            else
                Gfe(ig).str = -G_fe_mTpm ;
                k2t(peline, nfe:-1:1) = Gfe(ig).tst + kvoddt ;
            end
            
            Gpe(ig).tst = Gfe(ig).tfn ;
            Gpe(ig).tfn = Gpe(ig).tst + seq.blip_us ;
            
            Gpe(ig).str = -G_pe_step_mTpm ;
            
            Gpe(ig).A = (Gpe(ig).tfn - Gpe(ig).tst)*Gpe(ig).str;
            Gfe(ig).A = (Gfe(ig).tfn - Gfe(ig).tst)*Gfe(ig).str;
            
            Gpe(ig).Ast = Gpe(ig-1).Ast + Gpe(ig-1).A ;
            Gfe(ig).Ast = Gfe(ig-1).Ast + Gfe(ig-1).A ;
        end
        % One too many PE blips (one less than number of readouts)
        Gpe(ngfe) = [] ;
        
    otherwise
        error(['Unknown sequence model'])
end

seq.Gpe = Gpe ;
seq.Gfe = Gfe ;

seq.G_fe_mTpm  = G_fe_mTpm ;
seq.G_pe_step_mTpm = G_pe_step_mTpm ;

seq.k2t = k2t ;
seq.Tacq_us = Tacq_us;

end

function [gobj, gobj_last] = getgobj(G, currt)
% GETGOBJ Get gradient object number at specified time in sequence
% [gobj, gobj_last] = getgobj(G, currt)
%
% G is gradient structure with fields tst and tfn
% currt is time in sequence in us.
% gobj is the object number (may be empty if no object on at currt).
%
% D.Atkinson@ucl.ac.uk
%
%

Gt = [G.tst] - currt ;
locst = find(Gt<0) ;

Gt = [G.tfn] - currt ;
locfn = find(Gt<0) ;

gobj = setdiff(locst,locfn) ;

if isempty(locfn)
    gobj_last = [] ;
else
  gobj_last = locfn(end) ;
end

end

function plot_seq(seq)
% PLOT_SEQ Plots Sequence Gradients and Cumulative Area
% Assumes all gradients are rectangular
%
% plot_seq(seq)
%   seq is a structure with fields that include Gfe and Gpe
%
% David Atkinson  D.Atkinson@ucl.ac.uk
% See also 

Gfe = seq.Gfe ;
ngobj = length(Gfe) ;
cumAfe(1).tst = 0 ;

figure('Name','Gfe')
hold on, grid on
for igobj = 1:ngobj
  plot([ Gfe(igobj).tst Gfe(igobj).tst Gfe(igobj).tfn Gfe(igobj).tfn], ...
      [ 0 Gfe(igobj).str Gfe(igobj).str 0],'r')
  if igobj> 1
      cumAfe(igobj).tst = cumAfe(igobj-1).tfn ;
  end
  cumAfe(igobj).tfn = cumAfe(igobj).tst + Gfe(igobj).str * (Gfe(igobj).tfn-Gfe(igobj).tst) ;
  
  plot( [Gfe(igobj).tst  Gfe(igobj).tfn], [cumAfe(igobj).tst cumAfe(igobj).tfn],'g-')
end
 

Gpe = seq.Gpe ;
ngobj = length(Gpe) ;
cumApe(1).tst = 0 ;

for igobj = 1:ngobj
  plot([ Gpe(igobj).tst Gpe(igobj).tst Gpe(igobj).tfn Gpe(igobj).tfn], ...
      [ 0 Gpe(igobj).str Gpe(igobj).str 0],'b')
  
  if igobj> 1
      cumApe(igobj).tst = cumApe(igobj-1).tfn ;
  end
  cumApe(igobj).tfn = cumApe(igobj).tst + Gpe(igobj).str * (Gpe(igobj).tfn-Gpe(igobj).tst) ;
  
  plot( [Gpe(igobj).tst  Gpe(igobj).tfn], [cumApe(igobj).tst cumApe(igobj).tfn],'k-')
  
  if abs(cumApe(igobj).tfn) < 1e-10;
      disp(['Zero PE crossing at end of object ',num2str(igobj)])
  end
  
  
end



end

        
        
        
