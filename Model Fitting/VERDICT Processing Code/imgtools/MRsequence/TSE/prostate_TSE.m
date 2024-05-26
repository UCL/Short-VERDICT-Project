function [sigec, korder] = prostate_TSE(T1s, T2s)
% prostate_TSE.m
%
% Adapted from test4_multislice_TSE.m in Shaihan's EPG-X
%

%%% Sequence
%%% Number of refocusing pulses
nrefocus = 25;

%%% Common sequence parameters
ESP = 7.7;
TR = 5000; %<--- use this to work out time gap at the end
Ntr = 4; % DA moved this here
% set up korder, here interleaved TSE (not used in this function for signal
% estimation)
korder = [] ;
for itr = 1:Ntr
    korder = cat(1,korder,[itr:Ntr:Ntr*nrefocus]') ;
end

%%%  8 different slice_number scenarios in original code
% nslice = 1:2:15;
nslice = 1 ;
nn  = length(nslice);

%%% Generate refocusing pulse series
a0={};
a0{1} = d2r([90 180*ones(1,nrefocus)]);
a0{2} = d2r([90 160 120*ones(1,nrefocus-1)]);
nseq = length(a0) ;

npulse = nrefocus+1;
b1sqrdtau={};
b1sqrdtau{1} = [32.7 213.1*ones(1,nrefocus)]; % uT^2 ms
b1sqrdtau{2} = [36.7 189.4 106.5*ones(1,nrefocus-1)]; % uT^2 ms

%%% slice shifts are different for each expt
% DA I guess this refers to his brain measurements
df = [10.9 12.27] * 42.57e3 * 6e-3; % G * gamma * dx


sig={};
sigec= {} ;
tec = {} ;


%%% Loop over tissues
for itissue = 1:2
    
    %%% loop over the experiments (sequences)
    for iseq = 1:nseq
        
        z0 = {};
        mz = {};
        ss = {};
        
        for jsl = 1:nn %<-- loop over number of slices (each one was a separate experiment)
            
            %%% Set up sequence. Slice order goes odd then even.
            slice_order = [1:2:nslice(jsl) 2:2:nslice(jsl)];
            soi = ceil(nslice(jsl)/2);%<-- slice of interest is the middle slice
            
            %%% generate the lineshape for each slice (depends on frequency
            %%% offset)
            fs = df(iseq) * (-floor(nslice(jsl)/2):floor(nslice(jsl)/2));
            GG = interp1(ff,G,fs);
            
            Nsl = length(slice_order);
            
            slice_order = repmat(slice_order,[1 Ntr]);
            Ntot = Ntr * Nsl;
            
            %%% now work out delay to add after each slice
            Tshot = ESP*(nrefocus+0.5);
            Tdelay = TR/nslice(jsl) - Tshot;
            
            % how to evolve Z0 between slices
            L = [[-R1f-kf kb];[kf -R1b-kb]];
            C = [R1f*(1-f) R1b*f]';
            Xi = expm(L*Tdelay);
            I=eye(2);
            Zoff = (Xi - I)*inv(L)*C;
            
            %%% Initialise magnetization
            z0 = [(1-f) f];
            ss{jsl} = [];
            
            % loop over the slices/TRs
            
            for ii=1:Ntot  % this is seq_shot 
                seq_shot = ii ;
                
                if slice_order(ii)==soi
                    % Local slice
                    [s,Fn,Zn] = EPG_TSE(a0{iseq},ESP,T1s(itissue),T2s(itissue),'zinit',z0) ;
                    
                    % [s, Fn,Zn] = EPGX_TSE_MT(a0{iseq},b1sqrdtau{iseq},ESP,[1/R1f 1/R1b],1/R2f,f,kf,GG(soi),'zinit',z0);
                    % save signal only in this case
                    ss{jsl} = cat(2,ss{jsl},s(:));
                    
                else
                    error(['Should not be here'])
                    % Other slice
                    % set flips to zero and use the saturation lineshape for that slice
                    [s, Fn,Zn] = EPGX_TSE_MT(a0{iseq}*0,b1sqrdtau{iseq},ESP,[1/R1f 1/R1b],1/R2f,f,kf,GG(slice_order(ii)),'zinit',z0);
                end
                
                  
                % update z0
                %%% First take Z0 at end of TSE shot
                z0 = squeeze(Zn(1,end,:));
                
                %%% Now evolve it by amount due to recovery period
                z0 = Xi*z0 + Zoff;
                
                % t_echo = (seq_shot-1)*(Tshot+Tdelay) + jecho*ESP
                % so for the nrefocus echoes in s, the times are:
                % (seq_shot-1)*(Tshot+Tdelay) + [1:nrefocus]*ESP
                %
                % The slice is slice_order(ii)
                
                sigec{seq_shot, iseq, itissue} = abs(s(:)) ;
                tec{seq_shot, iseq, itissue} = (seq_shot-1)*(Tshot+Tdelay) + [1:nrefocus]'*ESP ;
                
                
            end
            sig{iseq}(jsl,itissue) = abs(ss{jsl}(13,end));%<- record signal in echo 13 of the last simulated TR period
            
            disp([iseq jsl itissue])
        end
        
    end
end

figure
iseq=1;
for ishot = 1:size(sigec,1)
    plot(tec{ishot,1,1}, sigec{ishot,iseq,1},'r')
    hold on
    grid on
    plot(tec{ishot,1,2}, sigec{ishot,iseq,2},'b')
    
    LWF = 0.2 ;
    plot(tec{ishot,1,2}, LWF*sigec{ishot,iseq,2} + (1-LWF)*sigec{ishot,iseq,1},'k')
    
    LWF = 0.1 ;
    plot(tec{ishot,1,2}, LWF*sigec{ishot,iseq,2} + (1-LWF)*sigec{ishot,iseq,1},'--k')
end
legend(legn)
title(['Refoc3: ',num2str(a0{iseq}(3)/(2*pi)*360)])

% sig{1}(:,3)=1; % CSF is defined as having no MT effect, so signals are same
% sig{2}(:,3)=1;

% % %% Load in images and ROI measurements (made in another script)
% % load bin/test4_imagedata
% % 
% % %% Assume in-vivo data was generated elsewhere - read in above
% % 
% % leg={'White Matter','Gray Matter (Caudate)','Cerebrospinal Fluid'};
% % 
% % figure(21)
% % clf
% % sl = [1 4 8];
% % nr=3;nc=3;
% % fs=18;
% % %%% 1st image, mark ROIs
% % subplot(nr,nc,1)
% % h1=imagesc(repmat(imrotate((abs(ims{1,1})-200)/1000,-90),[1 1 3]));% grayscale
% % hold on
% % for ii=1:3
% %     roitmp = imrotate(roi{ii},-90);
% %     roitmp = repmat(roitmp,[1 1 3]);
% %     roitmp(:,:,setdiff(1:3,ii))=0;
% %     h2 = imagesc(roitmp);
% %     set(h2, 'AlphaData', 0.5*imrotate(roi{ii},-90));
% % end
% % axis off
% % title('Single slice TSE','fontsize',fs)
% % 
% % for ii=2:nc
% %     subplot(nr,nc,ii)
% %     % DA old code: imsjm(ims{sl(ii),1},[200 1200],'rot',-90,'gray')
% %     % replaced with:
% %     imshow(rot90(ims{sl(ii),1},3),[200 1200])
% %     title(sprintf('Multislice (%d slices)',nslice(sl(ii))),'fontsize',fs)
% %     axis off
% % end


% for kk=1:2  % DA expts  180 or 160 120 ..
%     for ii=1:3  % DA tissue
%         subplot(2,1,kk)
%         % subplot(nr,nc,ii + nc*(kk-1)+nc)
%         hold on
%         grid on
%         pp=[];
%         yplot = sig{kk}(:,ii)/sig{kk}(1,ii) ;
%         plot(nslice, yplot,'-o')
% %         for jj=1:nn
% %             pp(jj)=plot(nslice(jj),sig{kk}(jj,ii)/sig{kk}(1,ii),'--o');
% %         end
%         %set(pp,'markersize',8,'markerfacecolor',[1 0 0])
% %         if exist('xm','var')
% %             %%% compute scaling 
% %             sf = mean(sig{kk}(:,ii)/sig{kk}(1,ii))/mean(xm{ii,kk});
% %             %pp2=errorbar(1:2:15,xm{ii,kk}/xm{ii,kk}(1),xs{ii,kk}/xm{ii,kk}(1));% in-vivo data
% %             pp2=errorbar(1:2:15,xm{ii,kk}*sf,xs{ii,kk}*sf);% in-vivo data
% %             set(pp2,'marker','.','markersize',8,'color',[0 0 0])
% %         end
%         ylim([0.45 1.15])
%         if kk==1
%             title(leg{ii})
%         else
%             xlabel('Number of slices')
%         end
%         ylabel('signal (au)')
%         set(gca,'fontsize',12,'ytick',0.5:0.1:1.1)
%         xlim([0 32])
%     end
% end
% legend(legn)
% %
% gc = get(gcf,'children');
% 
% axes(gc(6))
% text(-5,0.62,'180° pulses','rotation',90,'fontsize',18,'fontweight','bold')
% axes(gc(3))
% text(-5,0.62,'120° pulses','rotation',90,'fontsize',18,'fontweight','bold')
% 
% %
% %%% reposition
% ww=850;hh=650;
% setpospap([100 100 ww*0.9 hh*0.9])
% 
% set(gc(9),'Position',[0.025 0.55 0.3*([1 (ww/hh)])])
% set(gc(8),'Position',[0.35 0.55 0.3*([1 (ww/hh)])])
% set(gc(7),'Position',[0.675 0.55 0.3*([1 (ww/hh)])])
% 
% for ii=1:3
%     gc(ii).Position = [0.7-0.3*(ii-1) 0.07 0.23 0.18];
%     gc(ii+3).Position = [0.7-0.3*(ii-1) 0.3 0.23 0.18];
% end
% 
% print -dpng -r300 bin/Test4_fig1.png

