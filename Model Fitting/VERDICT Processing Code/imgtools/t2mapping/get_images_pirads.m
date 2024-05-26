function [files,IM,T2,ADC,V] = get_images_pirads(home_dir,desktop,diff,echo_number,b_val)
% GET_IMAGES_PIRADS
% Code from Will Devine
%
% [files,IM,T2,ADC,V] = get_images_pirads(home_dir,desktop,diff,echo_number,b_val)
%

cd(home_dir)
%% create list of filemanse, removing irrelevant filenames (such as '.' and '..')
a = dir;
files = {a.name};
files = files(contains(files,'Inn'));

V_map_names = {'fIC','fVASC','fEES','cellularity','R'};
V = nan([176,176,14,numel(V_map_names),numel(files)]);% may need to change

index = [];
for i = 1:numel(files)
    cd(home_dir)
    f = files{i};
    cd(f)
    cd('study')
    %% 32 echo
    if exist('32echo')
        cd('32echo')
        b = dir;
        filenames = {b.name};
        filenames = filenames(contains(filenames,'.dcm'));
        filenames = filenames{1};
        
        sprintf('--------------------------------------------------')
        disp(strcat('Computing T2: ',num2str(i),'/',num2str(numel(files))))
        

        dinfo{i} = dmfparse2(filenames);
        T2{i} = unique([dinfo{1}.EffectiveEchoTime]).*1e-3;
        D = double(d2mat(dinfo{i},{'frame'}));
        IM{i} = permute(reshape(D,[size(D,1),size(D,2),echo_number,size(D,3)./echo_number]),[1,2,4,3]);
    end
    
    %% diffusion
    cd(diff)
    cd('INNOVATE')
    cd(upper(strcat(f(1:3),'-',f(4:6),'-',f(7:9))));
    dd = dir;
    cd(dd(end).name)
    names = {'b90','b500','b1500','b2000','b3000'}; %
    DIFF = [];
    for j = 1:numel(names)
        b = dir;
        b = {b.name};
        b = b(contains(b,names{j}));
        cd(b{1})
        cd('DICOM')
        
        c = dir;
        filenames = {c.name};
        filenames = filenames{end};
        D = squeeze(double(dicomread(filenames)));
        D2 = permute(reshape(D,size(D,1),size(D,2),5,size(D,3)/5),[1,2,4,3]);
        DIFF(:,:,:,j) = D2(:,:,:,5)./D2(:,:,:,1);
        cd ..
        cd ..
    end
    ADC{i} = calcADC(DIFF,b_val);
    %% VERDICT
    cd(desktop)
    cd('VERDICT/INNOVATE')
    c = dir;
    filenames2 = {c.name};
	ff = contains(filenames2,(f(4:6)));
    if max(ff)
        filenames2 = filenames2{ff};
        cd(filenames2)

        d = dir;
        d = {d.name};
        d = {d{contains(d,'INN-')}};
        cd(d{1})

        d = dir;
        d = {d.name};
        cd(d{end})

        cd('RMAPS1')

        e = dir;
        e = {e.name};
        for k = 1:numel(V_map_names)
            v = load_nii(e{contains(e,strcat(V_map_names{k},'.nii'))});
            V(:,:,:,k,i) = rot90(v.img,3);
        end
    end
end
