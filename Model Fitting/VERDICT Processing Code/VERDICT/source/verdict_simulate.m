function [gndt, scheme, Y, t, opt] = verdict_simulate
% VERDICT_SIMULATE Simulates VERDICT data for testing
%
% [gndt, scheme, Y, t, opt] = verdict_simulate
%
% Example
%  [gndt, scheme, Y, t, opt] = verdict_simulate ;
%  tCell = namedargs2cell(t) ; optCell = namedargs2cell(opt) ;
%  [fIC, fEES, fVASC, R] = verdict_fit(scheme,Y, tCell{:}, optCell{:}) ;
%  % compare gndt.fIC and fIC
%
% Optionally add noise:
%  Ynoise = abs(addnoise(Y,'snr',20,'signal',1) );
%
% Copyright 2022. David Atkinson
%
% See also verdict_fit ball astrosticks sphereGPD


t.dEES = 2 ;
t.dIC = 2 ;
t.dVASC = 8 ;
t.Rs = 1:15 ;

opt.mask = [] ;


% nvox = 3 values here that will correspond to voxels in Y which will be
% shape [nvox 1 1 nscheme]

fIC   = [ 0.5 0.7 0.9 ]' ;
fVASC = [ 0.1 0.1 0.05 ]' ;
R     = [ 9   8   7   ]' ;
fEES  = 1 - (fIC+fVASC) ;
nvox = length(fIC) ;
rDist = cell([nvox 1]);

gndt.nvox = nvox ;
gndt.fIC = fIC;
gndt.fVASC = fVASC ;
gndt.R = R ;
gndt.fEES = fEES ;

%                      delta, DELTA,   bval
scheme(1) = fill_scheme( 3.9, 23.8,   90) ;
scheme(2) = fill_scheme(11.4, 31.3,  500) ;
scheme(3) = fill_scheme(23.9, 43.8,  1500) ;
scheme(4) = fill_scheme(14.4, 34.3,  2000) ;
scheme(5) = fill_scheme(18.9, 38.8,  3000) ;

figure('Name','Signals') ;
tiledlayout(nvox,1)

nscheme = length(scheme) ;
nr = length(t.Rs) ;

Y = zeros([nvox 1 1 nscheme]) ;

for ivox = 1:nvox
    nexttile, hold on
    % Normal distribution of radii with mean R(ivox)
    pr = pdf('Normal',t.Rs,R(ivox),1) ; % mean, sigma .
    pr = pr./sum(pr) ;
    rDist{ivox} = pr ;

    A = zeros([nscheme nr+2]) ; 
    sIC = zeros([nscheme nr]) ;
    sEES = zeros([nscheme 1]) ;
    sVASC = zeros([nscheme 1]) ;
    stot = zeros([nscheme 1]) ;

    for ischeme = 1:nscheme
        for ir = 1:nr
            sIC(ischeme,ir) = sphereGPD(scheme(ischeme).delta, scheme(ischeme).DELTA, ...
                scheme(ischeme).G, t.Rs(ir), t.dIC);
        end
        sEES(ischeme) = ball(scheme(ischeme).bval, t.dEES) ;
        sVASC(ischeme) = astrosticks(scheme(ischeme).bval, t.dVASC) ;
        stot(ischeme) = fIC(ivox).*(sIC(ischeme,:)*pr') + ...
            fEES(ivox)*sEES(ischeme) + fVASC(ivox)*sVASC(ischeme) ;

        A(ischeme,:) = [sIC(ischeme,:) sEES(ischeme) sVASC(ischeme) ] ;
        Y(ivox, 1, 1, ischeme) = stot(ischeme) ;

        plot(scheme(ischeme).bval, fIC(ivox).*sIC(ischeme,:)*pr','r*', 'MarkerSize',10)
        plot(scheme(ischeme).bval, fEES(ivox)*sEES(ischeme), 'b*', 'MarkerSize',10)
        plot(scheme(ischeme).bval, fVASC(ivox)*sVASC(ischeme), 'g*', 'MarkerSize',10)
    end

    plot([scheme.bval], stot,'k')
    xlabel('bval')
    grid on
    axis([0 3000 0 1])
    title(['R ',num2str(R(ivox)),' fIC ',num2str(fIC(ivox))])

end

gndt.rDist = rDist ;
gndt.A = A;

end


function scheme = fill_scheme(delta, DELTA, bval)
% FILL_SCHEME Fills scheme structure and checks parameters
%
% See also stejskal

scheme.delta = delta;
scheme.DELTA = DELTA ;

scheme.bval = bval ;

scheme.G = stejskal(delta,DELTA,bval=bval);

    
end