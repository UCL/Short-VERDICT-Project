function R = relaxTO5(tempC, tesla, verbose)
% RELAXTO5 Relaxation properties of TO5 phantom at 1.5T and 3T
% R = relaxTO5(tempC, tesla)
% R = relaxTO5(tempC, tesla, verbose)
%
% tempC is temperature in C
% tesla must be 3 or 1.5 (numeric)
% verbose {false} | true  if true will plot T1 ans T2 for gels
%
% Output R has fields T1 and T2
%
% Example
% R = relaxTO5(20, 1.5, true)
%
% 
% TO5 data from TO5 Eurospins Appendix III 3.0T, +/-3%
%   T1          T2
%   292	296	300	292	296	300  Temp in Kelvin
%
% David Atkinson  D.Atkinson@ucl.ac.uk
%

if nargin < 3
    verbose = false ;
end

temps = [292	296 	300	] ;

TO5_3T = [ ...
	200	223	249	             52	50	48
	299	334	372	             73	70	67
	296	331	368	            113	111	110
	463	516	574	             53	 49	 46
	450	502	559	             94	 89	 85
	444	496	552	            154	151	148
	604	674	750	             95	 89	 84
	596	666	741	            136	129	124
	754	841	935	            116	109	103
	745	831	924	            157	149	142
	903	 1007 1120	        137	128	121
	1448 1615 1796	        390	373	359
	966	 1078 1199	        224	212	203
	1034	1153	1282	167	156	148
	1160	1293	1437	214	201	191
	1276	1422	1581	204	190	180
	1262	1407	1563	184	171	161
	1415	1576	1750	174	161	151 ] ;

TO5_1p5T = [ ...
    199 221 245 52 50 48
    298 331 367 73 70 67
    295 329 365 113 111 110
    455 506 560 53 49 46
    446 496 550 94 89 85
    442 492 546 154 151 147
    597 664 736 95 89 84
    592 659 731 136 129 124
    745 828 918 116 108 102
    739 822 911 157 149 142
    892 992 1099 137 128 121
    1440 1603 1779 390 373 359
    959 1068 1184 223 212 203
    1023 1137 1260 167 156 148
    1149 1278 1416 214 201 191
    1261 1403 1555 204 190 180
    1246 1385 1535 183 171 161
    1392 1547 1714 174 161 151 ] ;


zeroC = 273.15 ;

if abs(tesla - 3 ) < 0.01
    R.T1 = interp1(temps,TO5_3T(:,1:3)',tempC+zeroC,'spline') ;
    R.T2 = interp1(temps,TO5_3T(:,4:6)',tempC+zeroC,'spline') ;
elseif abs(tesla - 1.5 ) < 0.01
    R.T1 = interp1(temps,TO5_1p5T(:,1:3)',tempC+zeroC,'spline') ;
    R.T2 = interp1(temps,TO5_1p5T(:,4:6)',tempC+zeroC,'spline') ;
else
    error(['Only for 3T or 1.5T'])
end

if verbose
    figure('Name','TO5 gel relaxation values')
    plot(R.T1,'DisplayName','T1'), hold on
    plot(R.T2,'DisplayName','T2')
    grid on 
    title(['Relaxation values at ',num2str(tempC),'C and ',...
        num2str(tesla),'T.'])
    legend

end

