function ernst_display(varargin)
% ERNST_DISPLAY Display Ernst curves of signal versus flip angle
% ernst_display
%
% David Atkinson  D.Atkinson@ucl.ac.uk
%

 


% T1s = [1500  ] ;
% T1s = [200 : 200: 2200 ] ;
% T1s = [ 800 1900 ]

prompt = {'Enter TR range (ms):', 'Enter T1 values (ms)', 'Enter max FA (deg)'};
dlg_title = 'Input TR and T1 ranges';
num_lines = 1;
def = {'4','[200:200:2200]','20'} ;

answer = inputdlg(prompt,dlg_title,num_lines,def);

%TR = str2num(answer{1}) ;
TRs = eval(answer{1}) ;
T1s = eval(answer{2}) ;
FAmax = eval(answer{3}) ;

FAs = [0:0.5:FAmax]' ;
FAs_deg = FAs ;
FAs = FAs * 2*pi/360 ;

SI = zeros([length(FAs) length(T1s) length(TRs)]) ;
SIn = zeros([length(FAs) length(T1s) length(TRs)]) ;

figure
co = [0,0,1; 
    0,0.5,0;
    1,0,0;
    0,0.75,0.75;
    0.75,0,0.75;
    0.25,0.25,0.25 ;
     0    0.4470    0.7410 ;
    0.8500    0.3250    0.0980  ;
    0.9290    0.6940    0.1250 ;
    0.4940    0.1840    0.5560 ;
    0.4660    0.6740    0.1880 ;
    0.3010    0.7450    0.9330 ;
    0.6350    0.0780    0.1840];
set(groot,'defaultAxesColorOrder',co)

ilstr = 1;
for itr = 1:length(TRs)
    for it1 = 1:length(T1s)
        T1 = T1s(it1) ;
        E = exp(-TRs(itr)/T1) ;
        
        SI(:,it1,itr) = sin(FAs).*((1-E)./(1-cos(FAs).*E)) ;
        SIn(:,it1,itr) = SI(:,it1)./max(SI(:,it1)) ;
        lstr{ilstr} = [num2str(TRs(itr)),'ms ',num2str(T1s(it1))] ;
        ilstr = ilstr + 1;
    end
end

% subplot(2,1,1)
hp=plot(repmat(FAs_deg,[1 length(T1s)*length(TRs)]),reshape(SI,[length(FAs)  length(T1s)*length(TRs)] )) ;
set(hp,'LineWidth',2)
legend(lstr)

xlabel('Flip Angle')
grid
    
% figure
% plot(repmat(FAs_deg,[1 length(T1s)]),SIn)
% legend(num2str(T1s'))
% title(['TR: ',num2str(TR)])
% xlabel('Flip Angle')
% grid


% subplot(2,1,2)
% Y = SI./repmat(sin(FAs),[1 length(T1s)]) ;
% X = SI./repmat(tan(FAs),[1 length(T1s)]) ;
% plot(X,Y,'o--')
% grid
%     
% for it1 = 1:length(T1s)
%     y = Y(:,it1);
%     x = X(:,it1) ;
%     soln = polyfit(x,y,1) ;
%     T1_est = -TR/log(soln(1)) ;
%     disp(['T1_est: ',num2str(T1_est),'. true value: ',num2str(T1s(it1))])
% end

    
end
