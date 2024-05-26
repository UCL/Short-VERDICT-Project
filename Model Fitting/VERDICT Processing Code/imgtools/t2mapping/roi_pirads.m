function [T,D,Tindex,Dindex,Tnames,Dnames] = roi_pirads(files,echo_number,home_dir)
% ROI_PIRADS
%
% [T,D,Tindex,Dindex,Tnames,Dnames] = roi_pirads(files,echo_number,home_dir)
%
% Code from Will Devine
%

%% prepare ROIs
cd(home_dir)
r = dir;
r = {r.name};
r = r(contains(r,'.nii'));

t_locs = contains(r,'t_'); % locate which of these are T2 ROIs
d_locs = contains(r,'d_');

index = nan([1,numel(r)]);
for i = 1:numel(r) % find which ROI corresponds to which map
    index(i) = find(contains(files,r{i}(1:3)));
end

Tindex = index(t_locs);
Dindex = index(d_locs);

rt = r(t_locs);
rd = r(d_locs);
%% load ROIs
Tnames = {}; % healthy or suspicious
Dnames = {};
T_n = {}; % numbers, not included in outputs
D_n = {};

for i = 1:numel(rt)
    nn = char(rt{i});
    l = load_untouch_nii(rt{i});
    l = rot90(double(l.img));
    n = permute(reshape(l,[size(l,1),size(l,2),echo_number,size(l,3)./echo_number]),[1,2,4,3]);
    n(n==0) = nan;
    m = squeeze(max(max(max(n))));
    echo_index = m==1;
    T{i} = repmat(squeeze(n(:,:,:,echo_index)),[1,1,1,echo_number]);
    Tnames = {Tnames{:},nn(6:end-4)};
    T_n = {T_n{:},nn(1:3)};
end

for i = 1:numel(rd)
    nn = char(rd{i});
    l = load_untouch_nii(rd{i});
    l = fliplr(rot90(double(l.img),3));
    l(l==0) = nan;
    D{i} = l;
    Dnames = {Dnames{:},nn(6:end-4)};
    D_n = {D_n{:},nn(1:3)};
end
Tnames = upper(char(Tnames));
Dnames = upper(char(Dnames));

%% Check that names match
kt = char(rt);
kd = char(rd);

kt = kt(:,1:3);
kd = kd(:,1:3);

if ~strcmp(kt,char(T_n)) || ~strcmp(kd,char(D_n))
    error('Indices do not match.')
end
