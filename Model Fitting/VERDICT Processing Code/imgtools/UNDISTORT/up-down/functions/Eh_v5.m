function [v] = Eh_v5(u,arg1,arg2)


Sk1=Eh_v4(u(1:end/2,:,:),arg1);
Sk2=Eh_v4(u(end/2+1:end,:,:),arg2);
v=(Sk1+Sk2)/2;

end
