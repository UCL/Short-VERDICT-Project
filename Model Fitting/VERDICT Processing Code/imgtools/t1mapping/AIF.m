function c = AIF(t,method)
% AIF Arterial Input Function (AIF) Currently only Parker model.
% c = AIF(t, method)
% c = AIF(t) default Parker method MRM56p993 (2006)
%
% t time in seconds, t can be scalar or matrix.
% c blood conc in mMol?, has the same size as t.
%
% Parker method MRM56p993 (2006)  DOI: 10.1002/mrm.21066
% --------------
% combination of two gaussians, a sigmoid and exponential function
% Note this is really only valid for specific bolus protocol defined in 
% paper of 0.1 mmol kg-1, at 3ml s-1 . They used a relaxivity of 4.5
% s-1mM-1 (note could be higher at 3T).
%
% Example
% t = [0:100] ; plot(t,AIF(t))
%
% David Atkinson  D.Atkinson@ucl.ac.uk
%

% amp = 2 element matrix for  amplitude of 1st and 2nd gaussian (mmol s)
% cent = 2 element matrix for  centres of 1st and 2nd gaussian ( s)
% sig =  2 element matrix of width of first gaussian
% A = amplitude of exponential  (mmol) 
% B = decay rate of exponential (s-1)  
% s   steepness of sigmoid function (s-1)
% T =  centre of s for sigmoid function (s) 
% time = N element matrix of time points (s)

if nargin < 2
    method = 'Parker' ;
end

szt = size(t) ;
t = t(:) ;

switch method
    case 'Parker'
        A1=0.809*60; % (mmol s) amplitude of first gaussian
        A2=0.330*60; % (mmol s) amplitude of second gaussian
        cent1=0.17046*60; % (s) centre of first gaussian
        cent2=0.365*60; % (s) centre of second gaussian
        sig1=0.0563*60; % (s) width of first gaussian
        sig2=0.132*60; % (s) width of second gaussian
        alpha =1.050; % (mmol) amplitude of exponential
        beta=0.1685/60; % (s-1) % decay rate of exponential
        s=38.078/60; % (s-1) steepness of sigmoid function
        tau=0.483*60; % (s) centre of s for sigmoid function
        sqrt2pi = sqrt(2*pi) ;
        
        G1 = A1/(sig1 * sqrt2pi) *exp(-((t-cent1).^2)/(2*sig1*sig1)) ;
        G2 = A2/(sig2 * sqrt2pi) *exp(-((t-cent2).^2)/(2*sig2*sig2)) ;
        
        esig = alpha*exp(-beta*t)./(1+exp(-s*(t-tau))) ;
        
        c = G1 + G2 + esig ;
        
        c = reshape(c,szt) ;
     case 'ParkerM'
        A1=0.5*0.809*60; % (mmol s) amplitude of first gaussian
        A2=0.25*0.330*60; % (mmol s) amplitude of second gaussian
        cent1=(0.1+0.17046)*60; % (s) centre of first gaussian
        cent2=(0.1+0.365)*60; % (s) centre of second gaussian
        sig1=0.0563*60; % (s) width of first gaussian
        sig2=0.132*60; % (s) width of second gaussian
        alpha =2*1.050; % (mmol) amplitude of exponential
        beta=0.1685/60; % (s-1) % decay rate of exponential
        s=38.078/60; % (s-1) steepness of sigmoid function
        tau=0.1*60+0.5*0.483*60; % (s) centre of s for sigmoid function
        sqrt2pi = sqrt(2*pi) ;
        
        G1 = A1/(sig1 * sqrt2pi) *exp(-((t-cent1).^2)/(2*sig1*sig1)) ;
        G2 = A2/(sig2 * sqrt2pi) *exp(-((t-cent2).^2)/(2*sig2*sig2)) ;
        
        esig = alpha*exp(-beta*t)./(1+exp(-s*(t-tau))) ;
        
        c = G1 + G2 + esig ;
        
        c = reshape(c,szt) ;
      case 'ParkerMearly'
        A1=0.5*0.809*60; % (mmol s) amplitude of first gaussian
        A2=0.25*0.330*60; % (mmol s) amplitude of second gaussian
        cent1=0.17046*60; % (s) centre of first gaussian
        cent2=0.365*60; % (s) centre of second gaussian
        sig1=0.0563*60; % (s) width of first gaussian
        sig2=0.132*60; % (s) width of second gaussian
        alpha =2*1.050; % (mmol) amplitude of exponential
        beta=0.1685/60; % (s-1) % decay rate of exponential
        s=38.078/60; % (s-1) steepness of sigmoid function
        tau=0.5*0.483*60; % (s) centre of s for sigmoid function
        sqrt2pi = sqrt(2*pi) ;
        
        G1 = A1/(sig1 * sqrt2pi) *exp(-((t-cent1).^2)/(2*sig1*sig1)) ;
        G2 = A2/(sig2 * sqrt2pi) *exp(-((t-cent2).^2)/(2*sig2*sig2)) ;
        
        esig = alpha*exp(-beta*t)./(1+exp(-s*(t-tau))) ;
        
        c = G1 + G2 + esig ;
        
        c = reshape(c,szt) ;
        
    otherwise
        error(['Unknown AIF method: ',method])
end



  