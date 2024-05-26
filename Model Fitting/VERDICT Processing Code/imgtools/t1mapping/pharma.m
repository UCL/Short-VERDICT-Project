function c = pharma(x, Stimes, model)
% PHARMA DCE pharmacokinetic model. Extended Tofts model concentration.
%  c = pharma(x, Stimes, model)
%
% David Atkinson  D.Atkinson@ucl.ac.uk
%
% See also C2S DCEfit
%
 nt = length(Stimes) ;
 switch model.DCEmethod
     case 'ETM'
         Ktrans = x(1) ;
         ve = x(2) ;
         vp = x(3) ;
         if length(x)==4
             tonset = x(4) ;
         else
             tonset = model.tonset ;
         end
         
         c = zeros([1 nt]) ;
         for itime = 1:nt
             tim = Stimes(itime) - tonset ;
             
             fhconvint = @(tau) toftcint(tau, model.AIFmethod, Ktrans, ve, tim)  ;
             
             c(itime) = vp*AIF(tim, model.AIFmethod) + Ktrans*quadgk(fhconvint,0,tim) ;
         end
         
     otherwise
         error(['Unknown pharma model.DCEmethod: ',model.DCEmethod])
 end
end

 
function cn = toftcint(tau,AIFmethod,Ktrans,ve,t) 
 % TOFTCINT Tofts model convolution integral
 %  cn = toftcint(tau,Ktrans,ve,t) 
 %  
 % tau will be vector values.
 %
 % Used by quadgk in pharma
 % See also PHARMA AIF
 
 kep = Ktrans / ve ;
 cn = AIF(tau,AIFmethod).*exp(-kep.*(t-tau));
end
 