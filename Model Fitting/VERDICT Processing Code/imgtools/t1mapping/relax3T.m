function R = relax3T(tissue, prop)
% RELAX3T Relaxation properties of tissue and TO5 phantom at 3T
% R = relax3T(tissue)
% R = relax3T(tissue, prop)
%
% Output R has fields T1, T2, ratio (sqrt (T2/T1) for bFFE contrast) and 
% thetamax (flip angle in degrees for max bFFE signal).
% 
% Input tissue can be 'liver', 'blood', 'kidney',  'skeletal muscle', 'TO5'
% For 'TO5', require prop.gel for gel number (1-18) and prop.tempC for
% temperature in Celsius. Typical scan room temp is 20C
% 
%
% Example
%  prop.gel = 5 ; prop.tempC = 20 ;
%  R = relax3T('TO5',prop) 
%
% TO5 data from TO5 Eurospins Appendix III 3.0T, +/-3%
%   T1          T2
%   292	296	300	292	296	300  Temp in Kelvin
%
% David Atkinson  D.Atkinson@ucl.ac.uk
%

temps = [292	296	300	] ;

TO5 = [ ...
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

zeroC = 273.15 ;

switch tissue
    case 'liver'
        R.T1 = 812 ;
        R.T2 = 42 ;  % Stanisz MRM
    case 'blood'
        R.T1 = 1900 ;
        R.T2 = 275 ; % Stanisz MRM
    case 'kidney'
        R.T1 = 1194 ; 
        R.T2 = 56 ; % Stanisz MRM
    case 'skeletal muscle'
        R.T1 = 1412 ;
        R.T2 = 50 ; % Stanisz MRM
    case 'subcutaneous fat'
        R.T1 = 365 ;
        R.T2 = 133 ; % ISMRM p450 2003
    case 'TO5'
        % see above for reference
        R.T1 = interp1(temps,TO5(prop.gel,1:3),prop.tempC+zeroC,'spline') ;
        R.T2 = interp1(temps,TO5(prop.gel,4:6),prop.tempC+zeroC,'spline') ;
    otherwise
        warning(['Tissue ',tissue,' not found.'])
end

%bFFE parameters
R.ratio = sqrt(R.T2/R.T1) ;
R.thetamax = acos((R.T1-R.T2)/(R.T1+R.T2)) /(2*pi) * 360 ;

end

