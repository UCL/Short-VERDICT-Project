function front
% FRONT Bring current figure to front
% front

cf = gcf ;
figure(cf)
disp(['Figure ',num2str(cf.Number),' (',cf.Name ,') brought to front.'])
