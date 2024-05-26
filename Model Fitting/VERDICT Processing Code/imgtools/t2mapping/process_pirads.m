function process_pirads(stage_start)
% PROCESS_PIRADS
%
% process_pirads(stage_start)
% 'stage_start' variable starts the process at the stage of the input
%
% Code from Will Devine

echo_number = 32;
b = [90,500,1500,2000,3000];

if exist('/Users/will/Library/Mobile Documents/com~apple~CloudDocs/PIRADS_all','dir')>0 % I use this 'if' because I use both my desktop and laptop
    home_dir = '/Users/will/Library/Mobile Documents/com~apple~CloudDocs/PIRADS_all'; % T2 images folder
else
    home_dir =  '/Users/WilliamDevine/Google Drive/PIRADS_all';
end

if exist('/Users/will/Google Drive/NIFTI_GRANT','dir')>0
    roi_dir = '/Users/will/Google Drive/NIFTI_GRANT'; % ROIs folder
else
    roi_dir =  '/Users/WilliamDevine/Google Drive/NIFTI_GRANT';
end

if exist('/Users/will/Library/Mobile Documents/com~apple~CloudDocs/Stages','dir')>0
    stage_dir = '/Users/will/Library/Mobile Documents/com~apple~CloudDocs/Stages'; % where I save my results in stages
else
    stage_dir =  '/Users/WilliamDevine/Library/Mobile Documents/com~apple~CloudDocs/Stages';
end

if exist('/Users/will/Desktop','dir')>0
    desktop = '/Users/will/Desktop'; % the desktop
else
    desktop =  '/Users/WilliamDevine/Desktop';
end

diff =  '/Users/WilliamDevine/B_vals_INNOVATE'; % diffusion images
%% STAGE 1 - get images
if stage_start<2
    [files,IM,T2,ADC,V] = get_images_pirads(home_dir,desktop,diff,echo_number,b);
    cd(stage_dir)
    save('Stage_1.mat','files','IM','T2','ADC','V','-v7.3')
else
    cd(stage_dir)
    load('Stage_1.mat')
end

%% STAGE 2 - prepare ROIs
if stage_start<3
    [T,D,Tindex,Dindex,Tnames,Dnames] = roi_pirads(files,echo_number,roi_dir); % loads T2 & ADC ROIs of all 'files' input
    cd(stage_dir)
    save('Stage_2.mat','T','D','Tindex','Dindex','Tnames','Dnames','-v7.3')
else
    cd(stage_dir)
    load('Stage_2.mat')
end

%% STAGE 3 - processing
if stage_start<4
    for i = 1:numel(Dindex)
        OUT_d{i} = ADC{Dindex(i)}.*D{i};
    end
    

    for i = 1:numel(Dindex(Dindex<52)) %% no verdict maps for later images
        OUT_v{i} = V(:,:,:,:,Dindex(i)).*repmat(D{i},[1,1,1,size(V,4)]);
    end
    
    for i = 1:numel(Tindex)
        O = LWI(IM{Tindex(i)}.*T{i},T2{Tindex(i)});
        OUT_t{i} = O;
    end

    cd(stage_dir)
    save('Stage_3.mat','OUT_t','OUT_d','OUT_v','-v7.3')
else
    cd(stage_dir)
    load('Stage_3.mat')
end
    
%% STAGE 4 - Linking PIRADS to ROIs
if stage_start<5
    if exist('/Users/will/Dropbox','dir')>0
        cd /Users/will/Dropbox
    else
        cd /Users/WilliamDevine/Dropbox
    end
    
    Tscores = nan(size(Tindex));
    Dscores = nan(size(Dindex));
    
    [x,txt] = xlsread('ROIs_July17.xls');
    scores = x(2:end,2:3);
    txt = char(txt(2:end,1));
    
    for i = 1:numel(Tindex) %% loop through all ROIs
        num =  char(files(Tindex(i))); %% which subject is this ROI from
        j = find(contains(string(txt),num(4:6))); %% find the table row that contains these PIRADS values
        if sum(~isnan(scores(j,:)))==2 %% if the table contains two PIRADS values
            if contains(Tnames(i,:),'HEALTHY') %% if the ROI has the name 'HEALTHY'
                Tscores(i) = scores(j,2);
            else %% if the ROI is not named 'HEALTHY'
                Tscores(i) = scores(j,1);
            end
        elseif sum(~isnan(scores(j,:)))==1 %% if the table contains one PIRADS value
            Tscores(i) = scores(j,1);
        else %% if the table contains no PIRADS values
            Tscores(i) = NaN;
        end
    end

    for i = 1:numel(Dindex)
        num =  char(files(Dindex(i)));
        j = find(contains(string(txt),num(4:6)));
        if sum(~isnan(scores(j,:)))==2
            if contains(Dnames(i,:),'HEALTHY')
                Dscores(i) = scores(j,2);
            else
                Dscores(i) = scores(j,1);
            end
        elseif sum(~isnan(scores(j,:)))==1
            Dscores(i) = scores(j,1);
        else
            Dscores(i) = NaN;
        end
    end
    
    cd(stage_dir)
    save('Stage_4.mat','Tscores','Dscores','-v7.3')
else
    cd(stage_dir)
    load('Stage_4.mat')
end

%% STAGE 5 - medians
if stage_start<6
    % T2 medians
    mT = nan(numel(Tindex),size(OUT_t{1},4));
    for i = 1:numel(Tindex)
        for j = 1:size(OUT_t{i},4)
            M = OUT_t{i}(:,:,:,j);
            mT(i,j) = nanmedian(M(:));
        end
    end

    % ADC medians
    mD = nan(numel(Dindex),size(OUT_d{1},4));
    for i = 1:numel(Dindex)
        for j = 1:size(OUT_d{i},4)
            M = OUT_d{i}(:,:,:,j);
            mD(i,j) = nanmedian(M(:));
        end
    end

    % Verdict medians
    mV = nan(numel(Dindex(Dindex<52)),size(OUT_v{1},4));
    for i = 1:numel(Dindex(Dindex<52))
        for j = 1:size(OUT_v{i},4)
            M = OUT_v{i}(:,:,:,j);
            mV(i,j) = nanmedian(M(:));
        end
    end
    cd(stage_dir)
    save('Stage_5.mat','mT','mD','mV','-v7.3')
else
    cd(stage_dir)
    load('Stage_5.mat')
end

%% STAGE 6 - stats

% % Correlation ADCvT2 and VERDICTvT2
% Vt = [];
% Vd = [];
% for i = 1:numel(Dindex)
%     Cc = ismember(Tindex,Dindex(i));
%     Dd = transpose(contains(string(Tnames),string(Dnames(i,:))));
%     Ee = ismember(Tscores,Dscores(i));
%     Ff = Cc&Dd&Ee;
%     if max(Ff)==1
%         Vd = [Vd,i];
%         Vt = [Vt,find(Ff)];
%     end
% end
% 
% 
% ADCvT2 = nan([size(mT,2),2]);
% VERDICTvT2 = nan([size(mT,2),size(mV,2),2]);
% for i = 1:size(mT,2)
%     [A,P] = corrcoef(mT(Vt,i),mD(Vd));
%     ADCvT2(i,1) = A(2,1);
%     ADCvT2(i,2) = P(2,1);
%     for j = 1:size(mV,2)
%         [B,Q] = corrcoef(mT(Vt(Vt<find(Tindex==52, 1 )),i),mV(Vd(Vd<find(Dindex==52, 1 )),j));
%         VERDICTvT2(i,j,1) = B(2,1);
%         VERDICTvT2(i,j,2) = Q(2,1);
%     end
% end
% 
% Vnames=Dnames(Dindex<52);
% for i = 1:size(mT,2)
%     for j = 1:size(mV,2)
%         [B,Q] = corrcoef(mT(Vt(Vt<find(Tindex==52, 1 ))&~strcmp(Vnames(i),'H'),i),mV(Vd(Vd<find(Dindex==52, 1 ))&~strcmp(Vnames(i),'H'),j));
%         VERDICTvT2_2(i,j,1) = B(2,1);
%         VERDICTvT2_2(i,j,2) = Q(2,1);
%     end
% end


%% split by subject T2



St = Tscores(~isnan(mT(:,8)))';
St_class = nan(size(St,1),3);
St_class(St==1,1)=0;
St_class(St==2,1)=0;
St_class(St==3,1)=1;
St_class(St==3,2)=0;
St_class(St==4,2)=1;
St_class(St==5,2)=1;
St_class(:,3) = St>2;

tbl = table(Tscores',Tindex',mT(:,9));
tbl.Properties.VariableNames = {'Score','Index','lwf'};

tbl123 = table(St_class(~isnan(St_class(:,1)),1),transpose(Tindex(~isnan(St_class(:,1)))),mT(~isnan(St_class(:,1)),9));
tbl345 = table(St_class(~isnan(St_class(:,2)),2),transpose(Tindex(~isnan(St_class(:,2)))),mT(~isnan(St_class(:,2)),9));
tbl12345 = table(St_class(~isnan(St_class(:,3)),3),transpose(Tindex(~isnan(St_class(:,3)))),mT(~isnan(St_class(:,3)),9));
tbl123.Properties.VariableNames = {'Score','Index','lwf'};
tbl345.Properties.VariableNames = {'Score','Index','lwf'};
tbl12345.Properties.VariableNames = {'Score','Index','lwf'};

%% split by subject diffusion

StD = Dscores(~isnan(mD))';
StD_class = nan(size(StD,1),3);
StD_class(StD==1,1)=0;
StD_class(StD==2,1)=0;
StD_class(StD==3,1)=1;
StD_class(StD==3,2)=0;
StD_class(StD==4,2)=1;
StD_class(StD==5,2)=1;
StD_class(:,3) = StD>2;

tbl = table(Dscores',Dindex',mD);
tbl.Properties.VariableNames = {'Score','Index','ADC'};

tbl123d = table(StD_class(~isnan(StD_class(:,1)),1),transpose(Dindex(~isnan(StD_class(:,1)))),mD(~isnan(StD_class(:,1))));
tbl345d = table(StD_class(~isnan(StD_class(:,2)),2),transpose(Dindex(~isnan(StD_class(:,2)))),mD(~isnan(StD_class(:,2))));
tbl12345d = table(StD_class(~isnan(StD_class(:,3)),3),transpose(Dindex(~isnan(StD_class(:,3)))),mD(~isnan(StD_class(:,3))));
tbl123d.Properties.VariableNames = {'Score','Index','ADC'};
tbl345d.Properties.VariableNames = {'Score','Index','ADC'};
tbl12345d.Properties.VariableNames = {'Score','Index','ADC'};


%% logistic regression T2
cv_no = 5;
D = {tbl123,tbl345,tbl12345};
for d = 1:numel(D) %% loop through tables
    data = D{d}; %% change to loop through
    un = unique(data.Index);
    cv_size = numel(un);
    cv_ind = [1,round([1:cv_no].*(cv_size/cv_no))];
    while 1
        un2 = un(randperm(numel(un)));
        aaa = nan([1,cv_no]);
        for i  = 1:cv_no 
            dataB{i} = data(ismember(data.Index,un2(cv_ind(i):cv_ind(i+1)-1)),:);
            aaa(i) = numel(unique(dataB{i}.Score));
        end
        if min(aaa)==2
            break;
        end
    end

    cvi = [1:cv_no];
    for i = 1:cv_no %% loop through folds, looking at each
        cvi2 = cvi(cvi~=i);
        data_train = vertcat(dataB{cvi2});
        data_test = dataB{i};
        mdl{d,i} = fitglm(data_train.lwf,data_train.Score,'Distribution','binomial','Link','logit');
        m_pval{d,i} = mdl{d,i}.Coefficients{2,4};
        pred{d,i} = predict(mdl{d,i},data_test.lwf);
        [X{d,i},Y{d,i},Tvalue{d,i},AUC{d,i}] = perfcurve(data_test.Score,pred{d,i},1,'cost',[0,sum(data_test.Score==1);sum(data_test.Score==0),0]);
        if AUC{d,i}<0.5
            [X{d,i},Y{d,i},Tvalue{d,i},AUC{d,i}] = perfcurve(data_test.Score,pred{d,i},0,'cost',[0,sum(data_test.Score==1);sum(data_test.Score==0),0]);
        end
        [~,mm] = min(sqrt((X{d,i}.^2)+((1-Y{d,i}).^2)));
        Ot{d,i} = [X{d,i}(mm),Y{d,i}(mm),Tvalue{d,i}(mm),sum(pred{d,i}>=Tvalue{d,i}(mm)),sum(pred{d,i}<Tvalue{d,i}(mm))];
        
%          % kruskal wallis on bootstrapped data
%         stats1{d,i} = bootstrp(1000,@(x) pcT(x,data_test.Score),pred{d,i});
    end
end

AUC2 = cell2mat(AUC);
mean_AUC = mean(AUC2,2);

mean_pval = mean(cell2mat(m_pval),2);

O2 = reshape(cell2mat(Ot),[numel(D),5,cv_no]);
mean_O = mean(O2,3);

%% logistic regression D
cv_no = 5;
D = {tbl123d,tbl345d,tbl12345d};
for d = 1:numel(D) %% loop through tables
    data = D{d}; %% change to loop through
    un = unique(data.Index);
    cv_size = numel(un);
    cv_ind = [1,round([1:cv_no].*(cv_size/cv_no))];
    while 1
        un2 = un(randperm(numel(un)));
        aaa = nan([1,cv_no]);
        for i  = 1:cv_no 
            dataB{i} = data(ismember(data.Index,un2(cv_ind(i):cv_ind(i+1)-1)),:);
            aaa(i) = numel(unique(dataB{i}.Score));
        end
        if min(aaa)==2
            break;
        end
    end

    cvi = [1:cv_no];
    for i = 1:cv_no %% loop through folds, looking at each
        cvi2 = cvi(cvi~=i);
        data_train = vertcat(dataB{cvi2});
        data_test = dataB{i};
        mdl{d,i} = fitglm(data_train.ADC,data_train.Score,'Distribution','binomial','Link','logit');
        m_pvald{d,i} = mdl{d,i}.Coefficients{2,4};
        predd{d,i} = predict(mdl{d,i},data_test.ADC);
        [Xd{d,i},Yd{d,i},Tvalued{d,i},AUCd{d,i}] = perfcurve(data_test.Score,predd{d,i},1,'cost',[0,sum(data_test.Score==1);sum(data_test.Score==0),0]);
        [~,mm] = min(sqrt((Xd{d,i}.^2)+((1-Yd{d,i}).^2)));
        Od{d,i} = [Xd{d,i}(mm),Yd{d,i}(mm),Tvalued{d,i}(mm),sum(predd{d,i}>=Tvalued{d,i}(mm)),sum(predd{d,i}<Tvalued{d,i}(mm))];
        
%         % kruskal wallis on bootstrapped data
%         stats2{d,i} = bootstrp(1000,@(x) pcD(x,data_test.Score),predd{d,i});
    end
end

%%% kruskal test on two sets of bootstrapped data above
% for i = 1:numel(D)
%     for j = 1:cv_no
%         p = kruskalwallis([stats1{i,j},stats2{i,j}]);
%         ppp{i,j} = p;
%     end
% end

AUC2d = cell2mat(AUCd);
mean_AUCd = mean(AUC2d,2);

mean_pvald = mean(cell2mat(m_pvald),2);

O2d = reshape(cell2mat(Od),[numel(D),5,cv_no]);
mean_Od = mean(O2d,3);
end
