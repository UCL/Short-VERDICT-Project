function machine_learning_study(stage)
% machine_learning_study(stage)
%
% Code from Will Devine
%

echo_number = 32;
% 
% if exist('/Users/will/Google Drive/INNOVATE_GRANT','dir')>0
%     T2_dir = '/Users/will/Google Drive/INNOVATE_GRANT';
% else
%     T2_dir =  '/Users/WilliamDevine/Desktop/INNOVATE_GRANT';
% end

if exist('/Users/will/Library/Mobile Documents/com~apple~CloudDocs/PIRADS_all','dir')>0
    T2_dir = '/Users/will/Library/Mobile Documents/com~apple~CloudDocs/PIRADS_all';
else
    T2_dir =  '/Users/WilliamDevine/Library/Mobile Documents/com~apple~CloudDocs/PIRADS_all';
end

if exist('/Users/will/Google Drive/INNOVATE_XX','dir')>0
    xx_dir = '/Users/will/Google Drive/INNOVATE_XX';
else
    xx_dir =  '/Users/WilliamDevine/Google Drive/INNOVATE_XX';
end

if exist('/Users/will/Google Drive/PIRADS_all','dir')>0
    diff_dir = '/Users/will/Google Drive/PIRADS_all';
else
    diff_dir =  '/Users/WilliamDevine/Google Drive/PIRADS_all';
end

if exist('/Users/will/Google Drive/VERDICT/INNOVATE','dir')>0
    diff_dir2 = '/Users/will/Google Drive/VERDICT/INNOVATE';
else
    diff_dir2 =  '/Users/WilliamDevine/Google Drive/VERDICT/INNOVATE';
end

if exist('/Users/will/Library/Mobile Documents/com~apple~CloudDocs/ROIs_T2_Will_Shonit_Grant','dir')>0
    roi_dir = '/Users/will/Library/Mobile Documents/com~apple~CloudDocs/ROIs_T2_Will_Shonit_Grant';
else
    roi_dir =  '/Users/WilliamDevine/Library/Mobile Documents/com~apple~CloudDocs/ROIs_T2_Will_Shonit_Grant';
end

if exist('/Users/will/Google Drive/NIFTI_GRANT','dir')>0
    roi_dir2 = '/Users/will/Google Drive/NIFTI_GRANT';
else
    roi_dir2 =  '/Users/WilliamDevine/Google Drive/NIFTI_GRANT';
end

if exist('/Users/will/Google Drive/Work','dir')>0
    stage_dir = '/Users/will/Google Drive/Work';
else
    stage_dir = ('/Users/WilliamDevine/Google Drive/Work');
end

if exist('/Users/will/Desktop','dir')>0
    desktop = '/Users/will/Desktop';
else
    desktop =  '/Users/WilliamDevine/Desktop';
end
%% STAGE 1 - get images
if stage < 2
    %% xx files
    [ffxx,xx_name,xxb0_name] = get_xx(xx_dir);
    
    
    %% ROIs
    [ROIt,indexT,gradeT,sliceT,fROIt] = roi_grant3(ffxx,echo_number,roi_dir); % produces all ROIs that match to the files input, along with an index of which input file it came from and what the grade is of that ROI
    
    [ROId,indexD,gradeD,sliceD,fROId] = roi_grant3(ffxx,1,roi_dir2,'diff');
    gradeDb = gradeD{2};
    indexDb = indexD{2};
    sliceDb = sliceD{2};
    ROIdb = ROId{2};
    fROIdb = fROId{2};
    
    gradeD = gradeD{1};
    indexD = indexD{1};
    sliceD = sliceD{1};
    ROId = ROId{1};
    fROId = fROId{1};
    
    type = ['H';'L'];
    
    % match ROIs
    TableT1 = [ffxx(indexT),strcat('00',string(sliceT')),type(double(~contains(string(gradeT),'BEN'))+1)]; %% T2 files with gleason scores
    TableT2 = [ffxx(indexDb),strcat('00',string(sliceDb')),type(double(~contains(string(gradeDb),'HEAL'))+1)]; %% T2 files with pirads
    TableD = [ffxx(indexD),strcat('00',string(sliceD')),type(double(~contains(string(gradeD),'HEAL'))+1)]; %% diffusion files with pirads
    
    ResultT = [];
    for i = 1:size(TableT2)
        m = min(contains(TableD(:,[1,3]),TableT2(i,[1,3]))');
        if max(m)
            ResultT = [ResultT;TableT2(i,:)];
        end
    end
    fROI = unique(ResultT(:,1));
    
    
    TT = strcat(ResultT(:,1),'d_',ResultT(:,3));
    TT = repmat(TT,[1,2]);
    TT(:,2) = strrep(TT(:,2),'_L','_S');
    TTb = strcat(ResultT(:,1),'t_',ResultT(:,3));
    TTb = repmat(TTb,[1,2]);
    TTb(:,2) = strrep(TTb(:,2),'_L','_S');
    
    ROId = ROId(contains(fROId,TT));
    indexD = indexD(contains(fROId,TT));
    gradeD = gradeD(contains(fROId,TT));
    sliceD = sliceD(contains(fROId,TT));
    fROId = fROId(contains(fROId,TT));
    
    ROIdb = ROIdb(contains(fROIdb,TTb));
    indexDb = indexDb(contains(fROIdb,TTb));
    gradeDb = gradeDb(contains(fROIdb,TTb));
    sliceDb = sliceDb(contains(fROIdb,TTb));
    fROIdb = fROIdb(contains(fROIdb,TTb));
    
    %% IMAGES
    cd(T2_dir)
    a = dir;
    a = {a.name};
    a = char(a(~contains(a,'.')));
    fT = string(a(:,4:6));
    
    cd(diff_dir2)
    a = dir;
    a = {a.name};
    a = char(a(~contains(a,'.')));
    fD = string(a(:,5:7));
    fB0 = string(a(:,5:7));
    
    ff = intersect(fT,fD); % subjects that have both T2 and diffusion
    ff_overall = intersect(ff,ffxx); % subjects that have xx files too
    ff_overall = intersect(ff_overall,fROI); % subjects that have T2 and diffusion ROIs too
    
    manual_choice = ff_overall;
    not = ["077","078","079","082","083","084","085","087","088","089","092","094","095","105","106","107","110","113","115","119","122","124","125","140","144","152","175","177","179","180","189"]; % don't line up diff & T2
    manual_choice(contains(manual_choice,not))=[];
    ff_overall = intersect(ff_overall,manual_choice);
    
    [filesD,~,~,matpD,fnamed] = get_images3(diff_dir2,1,'diff',ff_overall);
    [filesT,IMt,T2,matpT2,fnamet] = get_images3(T2_dir,echo_number,'t2',ff_overall);
    [filesb0,~,~,matpB0,fnameb0] = get_images3(diff_dir2,1,'b0',ff_overall);  
    
    fnamed = fnamed(~cellfun(@isempty,filesD),:);
    filesD = filesD(~cellfun(@isempty,filesD));
    fnamet = fnamet(~cellfun(@isempty,filesT),:);
    filesT = filesT(~cellfun(@isempty,filesT));
    fnameb0 = fnameb0(~cellfun(@isempty,filesb0),:);
    filesb0 = filesb0(~cellfun(@isempty,filesb0));
    
    fnamed = fnamed(:,1:5);
    fnameb0(cellfun('isempty',fnameb0))='';
    fnamet(cellfun('isempty',fnamet))='';
    
    
    xx_name = xx_name(contains(string(xx_name(:,5)),strcat('INN-',ff_overall)),:);  
    xx_name(cellfun(@isempty,xx_name))={''};
    
    diffs = {"90","500","1500","2000","3000"};
    for i = 1:size(xx_name,1)
        for j = 1:numel(diffs)
            xxxn = xx_name(i,:);
            xxxn2 = xxxn(contains(xxxn,strcat(' b',diffs{j})));
            xx_name2{i,j} = xxxn2{1};
        end
    end
    
    xxb0_name = xxb0_name(contains(string(xxb0_name(:,1)),strcat('INN-',ff_overall)),1);
    
    for i = 1:numel(ff_overall)
        for j = 1:numel(diffs)
            fffd = fnamed(contains(string(fnamed),strcat('INN-',ff_overall{i})) & contains(fnamed,strcat('_b',diffs{j})));
            fffb0 = fnameb0(contains(string(fnameb0),strcat('INN-',ff_overall{i})));
            fnamed2{i,j} = fffd{1};
            fnameb02{i} = fffb0{1};
        end
    end

    cd(stage_dir)
    save('Stage_1.mat','-v7.3')
else
    cd(stage_dir)
    load Stage_1.mat -regexp ^(?!stage$|stage_dir$).
end
%% STAGE 2 - set up data
ROIt_names = cellfun(@(x) x(1:3), fROIdb(contains(fROIdb,ff_overall)), 'un', 0);
ROIt_imindex = nan([numel(ROIt_names),1]);
for i = 1:numel(ROIt_names)
    ROIt_imindex(i) = find(contains(filesT,ROIt_names{i}));
end

if stage < 3
    remove_roi = ["081","3";"100","1";"108","4";"143","2"];
    
    TT = TT(contains(TT,ff_overall));
    fROId = fROId(contains(fROId,TT));
    sliceROId = sliceD(contains(fROId,ff_overall));
    
    indexDb = indexDb(contains(fROIdb,ff_overall));
    ROIdb2 = ROIdb(contains(fROIdb,ff_overall));
    gradeDb2 = gradeDb(contains(fROIdb,ff_overall));
    sliceROIt = sliceDb(contains(fROIdb,ff_overall));
    
    slice_resliced = [];
    k=0;
    for i = 1:size(IMt,1)
        for j = 1:sum(contains(fROIdb(contains(fROIdb,ff_overall)),filesT{i}(4:6)))
            k=k+1;
            
            ROIdb_pre = zeros(size(IMt{i,1}));
            ROIdb_pre(:,:,sliceROIt(k),:) = repmat(ROIdb2{k}(:,:,~isnan(max(max(ROIdb2{k})))),[1,1,1,32]);
            ROIdb_pre(isnan(ROIdb_pre))=0;
        end
    end
    
    for i=length(remove_roi):-1:1
        rm_ind = contains(filesT(ROIt_imindex),remove_roi(i,1)) & contains(string(sliceROIt),remove_roi(i,2));
        sliceROId(rm_ind)=[];
        ROIdb2(rm_ind)=[];
        sliceROIt(rm_ind)=[];
%         ROIdb_resliced(rm_ind)=[];
%         slice_resliced(rm_ind)=[];
        ROIt_imindex(rm_ind)=[];
        gradeDb2(rm_ind)=[];
        indexDb(rm_ind)=[];
        fROId(rm_ind)=[];
    end
    cd(stage_dir)
    save('Stage_2.mat','-v7.3') 
else
    cd(stage_dir)
    load Stage_2.mat -regexp ^(?!stage$|stage_dir$).
end
%% STAGE 3 - Register Images
if stage < 4  
    % REGISTER IMAGES
    [optimizer,~] = imregconfig('multimodal');
    optimizer.InitialRadius = optimizer.InitialRadius/1.5;
    for i = 1:numel(filesD)
        [Iwarped{i}, vtowarp{i}, vstrucatwarp{i}] = b0c_test2(fnameb02{i}, xxb0_name{i}, fnamed2(i,:), xx_name2(i,:),1:14); %unwarping
        [vol{i}, matp_rs{i}] = dreslice(Iwarped{i}, matpD{i}, matpT2{i}); % reslicing
        
        cd(stage_dir)
        save('Stage_3.mat','-v7.3') 
    end

    cd(stage_dir)
    save('Stage_3.mat','-v7.3') 
else
    cd(stage_dir)
    load Stage_3.mat -regexp ^(?!stage$|stage_dir$|indexDb$).
end

%% STAGE 4 - ROI ML
if stage < 5 
    % line up PIRADS scores to ROIs
    if exist('/Users/will/Dropbox','dir')>0
        cd /Users/will/Dropbox
    else
        cd /Users/WilliamDevine/Dropbox
    end
    
    [x,txt] = xlsread('ROIs_July17.xls');
    x = x(contains(string(txt(:,1)),ff_overall),:);
    txt = txt(contains(string(txt(:,1)),ff_overall),1);
    scores = x(:,2:3);
   
    Tscores = nan([1,numel(ROIdb2)]);
    
    for i = 1:numel(Tscores) %% loop through all ROIs
        num =  fROId{i}(1:3); %% which subject is this ROI from
        j = find(contains(txt,num)); %% find the table row that contains these PIRADS values
        if sum(~isnan(scores(j,:)))==2 %% if the table contains two PIRADS values
            if contains(gradeDb2(i),'H') %% if the ROI has the name 'HEALTHY'
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
    
    
    % split ROI into patches
%     [~,~,T_ind] = unique(cellstr(gradeD));
    T_ind = Tscores>2;
    b = [1,1]; %% will split the ROI into bxb patches
    B = nan([numel(ROIt_imindex),floor(size(ROIdb2{1},1)/b(1)),floor(size(ROIdb2{1},2)/b(2)),2+size(ROIdb2{1},3)+(size(vol{1},4)*size(vol{1},5))]);
    for i = 1:numel(ROIt_imindex)
        z_val = sliceROIt(i);
        R = ROIdb2{i}(:,:,1);
        [r,c] = ind2sub(size(R),find(~isnan(R)));
        rr = ceil((max(r)-min(r))/b(2));
        cc = ceil((max(c)-min(c))/b(1));
        Pt = squeeze(IMt{ROIt_imindex(i)}(:,:,z_val,:)).*repmat(ROIdb2{i}(:,:,1),[1,1,size(ROIdb2{1},3)]);
        pp = reshape(vol{ROIt_imindex(i)}(:,:,z_val,:,:),size(vol{1},1),size(vol{1},2),[]);
        Pd = pp.*repmat(ROIdb2{i}(:,:,1),[1,1,size(pp,3)]);
        for x = 1:cc
            for y = 1:rr % split up into groups of pixels
                Qd = reshape(squeeze(Pd(min(r)+((y-1)*b(1)):min(r)+(y*b(1))-1,min(c)+((x-1)*b(2)):min(c)+(x*b(2))-1,:)),[b(1)*b(2),size(Pd,3)]);
                Qt = reshape(squeeze(Pt(min(r)+((y-1)*b(1)):min(r)+(y*b(1))-1,min(c)+((x-1)*b(2)):min(c)+(x*b(2))-1,:)),[b(1)*b(2),size(Pt,3)]);
                B(i,:,:,1) = indexDb(i);
                B(i,:,:,2) = T_ind(i);
                B(i,x,y,3:34) = nanmedian(Qt,1);
                B(i,x,y,35:end) = nanmedian(Qd,1);
            end
        end
    end
    BB = reshape(B,size(B,1)*size(B,2)*size(B,3),[]);
    BB = BB(~isnan(BB(:,3)),:);
    
    cd(stage_dir)
    save('Stage_4.mat','-v7.3')
else
    cd(stage_dir)
    load 'Stage_4.mat' -regexp ^(?!stage$|stage_dir$).
end   
    
%%  STAGE 5 - Neural-nets
if stage<6
    cv_no = 5;
    perm = 5;
    
    layer_max = 4;
    layer_min = 1;
    node_max = 61;
    node_min = 1;
    node_step = 5;
    
    [NNb,~] = split_data(BB,cv_no);
    
    cvi = 1:cv_no;
    for i = 1:cv_no %% loop through folds, looking at each
        cvi2 = cvi(cvi~=i);
        % T2 & diff
        NN_train{i} = cell2mat(cellfun(@transpose,NNb(cvi2),'UniformOutput',false));
        NN_test{i} = NNb{i}';
        % T2 only
        NN_train2{i} = NN_train{i}(1:34,:);
        NN_test2{i} = NN_test{i}(1:34,:);
        % diff only
        NN_train3{i} = NN_train{i}([1:2,35:end],:);
        NN_test3{i} = NN_test{i}([1:2,35:end],:);
        % T2 mono
        f_train = calc_mono_T2(NN_train{i},T2{1});
        f_test = calc_mono_T2(NN_test{i},T2{1});
        NN_train4{i} = [NN_train{i}(1:2,:);f_train];
        NN_test4{i} = [NN_test{i}(1:2,:);f_test];
        % ADC
        a_train = calc_mono_diff(NN_train{i},double([diffs{:}]));
        a_test = calc_mono_diff(NN_test{i},double([diffs{:}]));
        NN_train5{i} = [NN_train{i}(1:2,:);a_train];
        NN_test5{i} = [NN_test{i}(1:2,:);a_test];
        % T2 mono + ADC
        af_train = [calc_mono_T2(NN_train{i},T2{1});calc_mono_diff(NN_train{i},double([diffs{:}]))];
        af_test = [calc_mono_T2(NN_test{i},T2{1});calc_mono_diff(NN_test{i},double([diffs{:}]))];
        NN_train6{i} = [NN_train{i}(1:2,:);af_train];
        NN_test6{i} = [NN_test{i}(1:2,:);af_test];
    end
    
    for l = layer_max:-1:layer_min
        for n = node_max:-1*node_step:node_min
            neur = repmat(n,[1,l]);
            
            learning_curve(cell2mat(NN_train),[20,20,20,20],cv_no)
            learning_curve(cell2mat(NN_train2),[20,20,20,20],cv_no)
            learning_curve(cell2mat(NN_train3),[20,20,20,20],cv_no)
            learning_curve(cell2mat(NN_train4),[20,20,20,20],cv_no)
            learning_curve(cell2mat(NN_train5),[20,20,20,20],cv_no)
            learning_curve(cell2mat(NN_train6),[20,20,20,20],cv_no)
            
            [pred{l,n},tr{1},net{1}] = train_nn(NN_train,NN_test,cv_no,neur);
            [pred2{l,n},tr{2},net{2}] = train_nn(NN_train2,NN_test2,cv_no,neur);
            [pred3{l,n},tr{3},net{3}] = train_nn(NN_train3,NN_test3,cv_no,neur);
            [pred4{l,n},tr{4},net{4}] = train_nn(NN_train4,NN_test4,cv_no,neur);
            [pred5{l,n},tr{5},net{5}] = train_nn(NN_train5,NN_test5,cv_no,neur);
            [pred6{l,n},tr{6},net{6}] = train_nn(NN_train6,NN_test6,cv_no,neur);
        end
    end
    cd(stage_dir)
    cd('NN')
    save(strcat('all',num2str(neur),string(datetime)),'-v7.3')
%%
    PRED{1} = pred;
    PRED{2} = pred2;
    PRED{3} = pred3;
    PRED{4} = pred4;
    PRED{5} = pred5;
    PRED{6} = pred6;
    
    for p = 1:sum(contains(who,'pred'))
        X{p} = nan([layer_max-layer_min+1,1+((node_max-node_min)/node_step),cv_no,3]);
        Y{p} = nan([layer_max-layer_min+1,1+((node_max-node_min)/node_step),cv_no,3]);
        AUC{p} = nan([layer_max-layer_min+1,1+((node_max-node_min)/node_step),cv_no]);
        O{p} = nan([layer_max-layer_min+1,1+((node_max-node_min)/node_step),cv_no,2]);
        acc{p} = nan([layer_max-layer_min+1,1+((node_max-node_min)/node_step),cv_no]);
        for l = layer_min:layer_max
            for n = node_min:node_step:node_max
                for i = 1:cv_no
                    ll = l-layer_min+1;
                    nn = ((n-node_min)/node_step)+1;
                    if max(double(PRED{p}{l,n}{i}))-min(double(PRED{p}{l,n}{i}))~=0
                        [X{p}(ll,nn,i,:),Y{p}(ll,nn,i,:),~,AUC{p}(ll,nn,i),O{p}(ll,nn,i,:)] = perfcurve(NNb{i}(:,2),double(PRED{p}{l,n}{i})',1,'cost',[0,sum(round(PRED{p}{l,n}{i})==0);sum(round(PRED{p}{l,n}{i})==1),0]);
                    else
                        X{p}(ll,nn,i,:) = 0;
                        Y{p}(ll,nn,i,:) = 0;
                        AUC{p}(ll,nn,i) = 0;
                        O{p}(ll,nn,i,:) = 0;    
                    end
                    if AUC{p}(ll,nn,i) < 0.5
                        AUC{p}(ll,nn,i) = 1-AUC{p}(ll,nn,i);
                    end
                    acc{p}(ll,nn,i) = sum(round(PRED{p}{l,n}{i}')==NNb{i}(:,2))/numel(NNb{i}(:,2));
    %                 performance(l,nn,i) = perform(net,NN_test{i}(:,2)',pred{i});
                end
            end
        end
        AUC_tot{p} = mean(AUC{p},numel(size(AUC{p})));
        acc_tot{p} = mean(acc{p},numel(size(acc{p})));
        sens{p} = mean(O{p},3);
        
        AUC_tot{p} = AUC_tot{p}(:,2:end);
        acc_tot{p} = acc_tot{p}(:,2:end);
        sens{p} = sens{p}(:,2:end,:);
    end
    
    AUC_max = [];
    acc_max = [];
    sens_max = [];
    for i = 1:sum(contains(who,'pred'))
        [a,b] = max(AUC_tot{i}(:));
        [rr,cc] = ind2sub(size(AUC_tot{i}),b);
        AUC_max = [AUC_max,a];
        acc_max = [acc_max,acc_tot{i}(rr,cc)];
        sens_max = [sens_max,squeeze(sens{i}(rr,cc,:))];
    end
    for f = 1:numel(sliceROIt)
        plot_map(net{2},IMt{ROIt_imindex(f)}(:,:,sliceROIt(f),:),vol{ROIt_imindex(f)}(:,:,sliceROIt(f),:,:),ROIdb2{f}(:,:,~isnan(max(max(ROIdb2{1})))),gradeDb2(f))
    end
    
    %% voxel-wise ML
    T2_table = [];
    for i = 1:numel(indexT)
        T_mat = reshape(IMt{indexT(i)}.*ROIt{i},[size(ROIt{i},1)*size(ROIt{i},2)*size(ROIt{i},3),size(ROIt{i},4)]); % change here to use patches rather than individual voxels
        T_mat = T_mat(~isnan(T_mat(:,1)),:);
        T_mat = [ones([size(T_mat,1),1]).*T_ind(i),T_mat];
        T2_table = [T2_table;T_mat];
    end
    cd(desktop)
    cd('ML_RESULTS')
    csvwrite('ml_table_pixel.csv',T2_table)
    msg = {strcat('1=',string(gradeT(find(T_ind==1,1),:))) strcat('2=',string(gradeT(find(T_ind==2,1),:))) strcat('3=',string(gradeT(find(T_ind==3,1),:))) strcat('4=',string(gradeT(find(T_ind==4,1),:)))};
    h = msgbox(msg,'Gleason scores');

    cd(stage_dir)
    save('Stage_5.mat','mT','mD','Vt','Vd','T2_table','BB','msg','-v7.3')
else
    cd(stage_dir)
    load('Stage_5.mat')
end

end
function [M,mn,sn] = normalise_inputs(varargin)
    N = varargin{1};
    if nargin>1
        mn = varargin{2};
        sn = varargin{3};
    else
        mn = mean(N(3:end,:),2);
        sn = std(N(3:end,:),0,2);
    end
    M = [N(1:2,:);(N(3:end,:)-mn)./sn];
end

function [ffxx,xx_name,xxb0_name] = get_xx(xx_dir)
    cd(xx_dir)
    cd('DICOM')
    a = dir;
    a = {a.name};
    a = a(contains(a,'INN'));
    for i = 1:numel(a)
        cd(a{i})
        b = dir;
        b = {b.name};
        b = b{~contains(b,'.')};
        cd(b)
        c = dir;
        c = {c.name};
        c = c(contains(c,' b'));
        for j = 1:numel(c)
            cd(c{j})
            d = dir;
            d = {d.name};
            d = d{contains(d,'.dcm')};
            xx_name{i,j} = strcat(xx_dir,'/DICOM/',a{i},'/',b,'/',c{j},'/',d);
            cd ..
        end
        c = dir;
        c = {c.name};
        c = c(contains(c,'B0'));
        for j = 1:numel(c)
            cd(c{j})
            d = dir;
            d = {d.name};
            d = d{contains(d,'.dcm')};
            xxb0_name{i,j} = strcat(xx_dir,'/DICOM/',a{i},'/',b,'/',c{j},'/',d);
            cd ..
        end
        
        cd ..
        cd ..
    end
    xx_nums = char(xx_name(~cellfun(@isempty,xx_name(:,5)),1));
    xxb0_nums = char(xxb0_name(~cellfun(@isempty,xxb0_name(:,1)),1));
    
    if exist('/Users/WilliamDevine/Desktop','dir')>0
        xx_nums = string(xx_nums(:,57:59));
        xxb0_nums = string(xxb0_nums(:,57:59));
    else
        xx_nums = string(xx_nums(:,48:50));
        xxb0_nums = string(xxb0_nums(:,48:50));
    end
    ffxx = intersect(xx_nums,xxb0_nums);
end

function [NNb,aaa] = split_data(BB,cv_no)
    un = unique(BB(:,1));
    cv_size = numel(un);
    cv_ind = [1,round([1:cv_no].*(cv_size/cv_no))];
    while 1
        un2 = un(randperm(numel(un)));
        aaa = nan([1,cv_no]);
        for i  = 1:cv_no 
            NNb{i} = BB(ismember(BB(:,1),un2(cv_ind(i):cv_ind(i+1)-1)),:);
            aaa(i) = numel(unique(NNb{i}(:,2)));
        end
        if min(aaa)==2
            break;
        end
    end
end

function NNb = split_data2(BB,cv_no)
    e = [1,round([1:cv_no].*(size(BB,1)/cv_no))];
    for i  = 1:cv_no 
        NNb{i} = BB(e(i):e(i+1)-1,:);
    end
end

function [pred,tr,net2] = train_nn(IN_train,IN_test,cv_no,neur)
    for i = 1:cv_no %% loop through folds, looking at each
        [NN_train,mval,vval] = normalise_inputs(IN_train{i});
        [NN_test{i},~,~] = normalise_inputs(IN_test{i},mval,vval);
        NN_train = NN_train';
        NN_test{i} = NN_test{i}';
        net = patternnet(neur);
        
        net.divideparam.trainRatio = 0.75;
        net.divideparam.valRatio = 0.25;
        net.divideparam.testRatio = 0;
        [net1,tr] = train(net,NN_train(:,3:end)',NN_train(:,2)');
        pred{i} = (net1(NN_test{i}(:,3:end)'))>0.5;
        net2{i} = net1;
    end
end

function OUT = calc_mono_T2(NN,T2)
    OUT = nan([1,size(NN,2)]);
    for i = 1:size(NN,2)
        f = log(NN(3:34,i)./NN(3,i));
        OUT(i) = -1./(T2(~isinf(f'))'\f(~isinf(f)));
    end
end

function OUT = calc_mono_diff(NN,b)
    IN = reshape(NN(35:end,:),numel(b),size(NN(35:end,:),1)/numel(b),[]);
    IN = squeeze(mean(IN(:,2:4,:),2)./IN(:,1,:));
    OUT = nan([1,size(IN,2)]);
    for i = 1:numel(OUT)
        f = log(IN(:,i)./NN(1,i));
        OUT(i) = -1*(b(~isinf(f'))'\f(~isinf(f)));
    end
end

function learning_curve(IN,neur,cv_no)
    iter=10;
    error_train = [];
    error_val = [];
    mse_t = nan([cv_no,iter]);
   	mse_v = nan([cv_no,iter]);
    
    net = patternnet(neur);
    
    net.divideparam.trainRatio = 1-(1/(cv_no-1));
    net.divideparam.valRatio = 1/(cv_no-1);
    net.divideparam.testRatio = 0;
    net.performFcn = 'mse';
    net.trainFcn = 'trainlm';
    
    IN_norm = normalise_inputs(IN)';
    NN_train = split_data(IN_norm,cv_no);
    for i = 1:cv_no
        nnt = [];
        nnv = [];
        
        NNt = cell2mat(NN_train(~ismember(1:cv_no,i))')';
        NNv = NN_train{i}';
        ITER = round((1:iter).*(size(NNt,2)/iter));
        for j=1:iter
            NNt_iter = NNt(:,1:ITER(j));
            [net1,tr] = train(net,NNt_iter(3:end,:),NNt_iter(2,:));
            
            mse_t(i,j) = immse(NNt_iter(2,:)',net1(NNt_iter(3:end,:))');
            mse_v(i,j) = immse(NNv(2,:)',net1(NNv(3:end,:))');
        end
    end
    error_train = mean(mse_t,1);
    error_val = mean(mse_v,1);
        
    figure
    plot(ITER,error_train)
    hold on
    plot(ITER,error_val)
    title(strcat(num2str(numel(neur)),' - ',num2str(neur(1))))
end   

function plot_map(net,t_img,d_img,ROI,grade)
    echo = 3;
    base_img = squeeze(t_img(:,:,:,3));
    
    image_t = squeeze(t_img);
    image_d = reshape(d_img,size(d_img,1),size(d_img,2),[]);
    
    image = image_t;
    image_ml = normalise_inputs(reshape(image,[size(image,1)*size(image,2),size(image,3)]));
    
    pred = nan([size(image(:,:,1)),numel(net)]);
    for i = 1:numel(net)
        pred_ml = net{i}(image_ml');
        pred(:,:,i) = reshape(pred_ml,[size(image,1),size(image,2)]);
    end
    pred_mean = mean(pred,3);
    
    B = bwboundaries(~isnan(ROI));
    BW = B{1};
    
    figure
    imagesc(base_img)
    colormap(contrast(base_img))
    hold on
    plot(BW(:,2),BW(:,1),'r')
    title(grade)
    
    figure
    imagesc(pred_mean)
    hold on
    plot(BW(:,2),BW(:,1),'r')
    title(grade)
    
    for i = 1:numel(net)
        figure
        imagesc(pred(:,:,i))
        hold on
        plot(BW(:,2),BW(:,1),'r')
        title(grade)
    end
        
end
