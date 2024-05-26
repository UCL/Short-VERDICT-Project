function plot_t1ffe

pref_group = 'plot_t1ffe' ;

FAs = [ 1:0.5:20] ;
nFA = length(FAs) ;
sd = zeros([ 1 nFA]) ;
ern = zeros([1 nFA]) ;

params = {'TR','T1','T2','flipphaseincrincr'} ;

defans = getpref(pref_group, params, {'10','1000','100','117'}) ;

answer = inputdlg(params ,'Input params', 1, defans) ;

[TRs, T1s, T2s, flipphaseincrincrs] = answer{:} ;
setpref(pref_group,params, answer) ;

TR = str2double(TRs)/1000 ;
T1 = str2double(T1s)/1000 ;
T2 = str2double(T2s)/1000 ;
flipphaseincrincr = str2double(flipphaseincrincrs) ;

for ifa = 1:nFA
    sdd = epg_t1ffe('FA',FAs(ifa), ...
        'flipphaseincrincr',flipphaseincrincr,'T1',T1,'TR',TR,'T2',T2) ;
    sd(ifa) = abs(sdd(end));
    ern(ifa) = ernstfn(T1, FAs(ifa), TR) ;
end
figure
plot(FAs,sd), hold on
plot(FAs,ern,'r')
xlabel('FA')
ylabel('SI')
grid on
title(['T1 ',num2str(T1*1000),'  TR ',num2str(TR*1000),...
    ' T2 ',num2str(T2*1000),' incr ',num2str(flipphaseincrincr)])
a = axis;
axis([0 FAs(end) 0 ceil(a(4)*100)/100 ])
