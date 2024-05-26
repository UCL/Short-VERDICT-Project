function [Mz, times] = calc_inversion(T1, ITR)
% calc_inversion
%
%
% Plot of Mz as a function of time for seqeunce with inversion pulses
% Can check null point against: 
% https://www.seichokai.or.jp/fuchu/dept1603.php
%
% D.Atkinson@ucl.ac.uk
%
%


dt = 1 ; % ms
dur = 10000 ;

% 0 1    2    3    4   5   ... dt
% 0  1.6   3.2  4.8  ...  inv times

times = [0: dt: dur] ;
nt = length(times) ;
Mz = zeros([1 nt]) ;

% start with inversion
M0 = 1; 
Mz(1) = -M0 ; 
invt = [0: ITR : dur] ;
ninv = length(invt) ;

next_inv = 2;

for it = 2:nt
   currt = times(it) ;
   if currt >= invt(next_inv) && next_inv <= ninv
       % evolve Mz up to inversion
       deltat = invt(next_inv) - times(it-1) ;
       Mz(it) = M0 + (Mz(it-1) - M0)*exp(-(deltat/T1)) ;
       
       % invert
       Mz(it) = -Mz(it-1) ; % update current value of Mz(it)
       
       % evolve to end of time period
       deltat = times(it) - invt(next_inv) ;
       Mz(it) = M0 + (Mz(it) - M0)*exp(-(deltat/T1)) ;
       
       next_inv = next_inv + 1 ;
   else
       % evolve over time period
       deltat = times(it) - times(it-1) ;
       Mz(it) = M0 + (Mz(it-1) - M0)*exp(-(deltat/T1)) ;
   end
   
end


   
end