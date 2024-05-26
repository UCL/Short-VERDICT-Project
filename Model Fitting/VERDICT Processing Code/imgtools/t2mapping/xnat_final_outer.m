function xnat_final_outer
% XNAT_FINAL_OUTER
%
% Code from Will Devine (previous extension was .mat)
%

% OUT='/Users/WilliamDevine/Desktop/test_dicomwrite';
% in='/Users/WilliamDevine/Library/Mobile Documents/com~apple~CloudDocs/PIRADS_all';

OUT='/Users/will/Desktop/test_dicomwrite';
in='/Users/will/Library/Mobile Documents/com~apple~CloudDocs/PIRADS_all';

cd(in)
a = dir;
files = {a.name};
files = files(contains(files,'Inn'));

for i = 1:numel(files)
    cd(in)
    f = files{i};
    cd(f)
    cd('study')
    %% 32 echo
    if exist('32echo')
        IN{i} = strcat(in,'/',f,'/study/32echo');
    end
end

for i=1:numel(IN)
    cd(IN{i})
    idx = strfind(IN{i},'Inn');
    out_folder_name = IN{i}(idx:idx+8);
    
    xnat_final(IN{i},OUT,out_folder_name);
end
    
