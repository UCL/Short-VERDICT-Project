function [cfa EVs cmap]=create_colour_FA(D,method,coord_system)
% Jack Harmer

[sy sx sz ~]=size(D);
r=zeros(sy,sx);
g=zeros(sy,sx);
b=zeros(sy,sx);

for cy=1:sy
    for cx=1:sx       
        switch method
            case 'EIG'
                DiffusionTensor=squeeze(D(cy,cx,1,:,:));
                [EigenVectors,EigenValues]=eig(DiffusionTensor); %maybe use SVD?
                
                r(cy,cx)=double((EigenVectors(1,3)));
                g(cy,cx)=double((EigenVectors(2,3)));
                b(cy,cx)=double((EigenVectors(3,3)));
            case 'SVD'
                [U,S,V] = svd(squeeze(D(cy,cx,1,:,:)),0);
                % U should be in LPS (DICOM)
                % want L red, P green, S blue ie no swaps needed
                r(cy,cx)=U(1);
                g(cy,cx)=U(2);
                b(cy,cx)=U(3);        
        end
    end
end

r_i=find(coord_system=='r');
g_i=find(coord_system=='g');
b_i=find(coord_system=='b');

cmap(:,:,r_i)=r;
cmap(:,:,g_i)=g;
cmap(:,:,b_i)=b;

EVs=cmap;

cmap=abs(cmap);

%calculate FA map
fa=invariantsb(D,'fa');

cfa=repmat(mat2gray(fa,[0 1]),[1 1 3]).*cmap;


end