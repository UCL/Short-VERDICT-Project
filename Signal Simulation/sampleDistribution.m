function signal = sampleDistribution(pdf, signals)

arguments
    pdf % signal pdf
    signals % array of signals (over which pdf is defined)
end

% == Method

% 1. Generate cumulative distribibution (cumtrapz)
% 2. Generate random decimal d = [0,1] (from uniform distribution)
% 3. Find signal for which F(s) = d
% 4. Return signal 


% ====================================================

% 1. Generate cumulative distribibution (cumtrapz)

F = cumtrapz(signals, pdf);


% 2. Generate random decimal d = [0,1] (from uniform distribution)
d = rand();

% 3. Find signal for which F(s) = d
[m,I] = min( abs(F - d) );

% 4. Return signal
signal = signals(I);


end