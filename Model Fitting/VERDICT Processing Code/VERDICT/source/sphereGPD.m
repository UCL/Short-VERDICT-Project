function sig = sphereGPD(delta, DELTA, G, r, d)
% SPHEREGPD Restricted Gaussian diffusion from a sphere
%
%  sig = sphereGPD(delta, DELTA, G, r, d)
%
% delta, DELTA timing parameters [ms]
% G gradient strength [mT/m]
% r radius [um]
% d diffusivity [1e-9 m^2/s] , typical value 2
% Values should be scalars.
%
% Adapted from Camino source reproduced at 
% https://git.fmrib.ox.ac.uk/fsl/DIVE/-/issues/28
%
% Copyright 2022. David Atkinson
%
% See also ball astrosticks stejskal verdict_fit

arguments
    delta (1,1) {mustBeNonnegative}
    DELTA (1,1) {mustBeNonnegative}
    G (1,1) {mustBeNonnegative}
    r (1,1) {mustBeNonnegative}
    d (1,1) {mustBeNonnegative}
end

% From CAMINO
%  60 first roots from the equation (am*x)j3/2'(am*x)- 1/2 J3/2(am*x)=0

am =[ 2.08157597781810, 5.94036999057271, 9.20584014293667, ...
    12.4044450219020, 15.5792364103872, 18.7426455847748, ...
    21.8996964794928, 25.0528252809930, 28.2033610039524, ...
    31.3520917265645, 34.4995149213670, 37.6459603230864, ...
    40.7916552312719, 43.9367614714198, 47.0813974121542, ...
    50.2256516491831, 53.3695918204908, 56.5132704621986, ...
    59.6567290035279, 62.8000005565198, 65.9431119046553, ...
    69.0860849466452, 72.2289377620154, 75.3716854092873, ...
    78.5143405319308, 81.6569138240367, 84.7994143922025, ...
    87.9418500396598, 91.0842274914688, 94.2265525745684, ...
    97.3688303629010, 100.511065295271, 103.653261271734, ...
    106.795421732944, 109.937549725876, 113.079647958579, ...
    116.221718846033, 116.221718846033, 119.363764548757, ...
    122.505787005472, 125.647787960854, 128.789768989223, ...
    131.931731514843, 135.073676829384, 138.215606107009, ...
    141.357520417437, 144.499420737305, 147.641307960079, ...
    150.783182904724, 153.925046323312, 157.066898907715, ...
    166.492397790874, 169.634212946261, 172.776020008465, ...
    175.917819411203, 179.059611557741, 182.201396823524, ...
    185.343175558534, 188.484948089409, 191.626714721361 ] ;


% gamma = 42.577478518*1e6     # [sec]^-1 * [T]^-1
gamma = 2.6752218744*1e8 ;
G_T_per_micron  = G*1e-3*1e-6   ;  % [T] * [um]^-1
gamma_ms        = gamma*1e-3 ;     % [ms]^-1 *[T]^-1
am1 = am/r ;
GPDsum = compute_GPDsum(am1,delta,DELTA,d,r) ;
log_att = -2*gamma_ms^2*G_T_per_micron^2*GPDsum ;
sig = exp(log_att) ;
end

function GPDsum = compute_GPDsum(am1,delta,DELTA,d,r)
    dam = d.*am1.*am1 ;
    e11 = -dam*delta ;
    e2  = -dam*DELTA ;
    dif = DELTA-delta ;
    e3  = -dam * dif ;
    plusd = DELTA+delta ;
    e4  = -dam*plusd ;
    nom = 2*dam*delta - 2 + (2*exp(e11)) + (2*exp(e2)) - exp(e3) - exp(e4) ;
    denom = dam.^2.*(am1.^2).*(r^2 * am1.^2 -2) ;
    GPDsum = sum(nom./denom) ;
end
