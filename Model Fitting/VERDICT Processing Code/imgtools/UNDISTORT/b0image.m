function [b0app, b0mod] = b0image(b0app, b0mod, col, opt)

ncr1 = length(opt.cr1) ;  % image coordinates

for icr1 = 1: ncr1
    b0app(icr1,col) = opt.b0fun(icr1) ;
    b0mod(icr1,col) = opt.b0funmodify(icr1) ;
end