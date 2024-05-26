function mip_rgb(rgb,sc, vdims)

ndsc = ndims(sc) ;
szsc = size(sc) ; ny = szsc(1); nx = szsc(2); nz=szsc(3);
szrgb = size(rgb) ;

if ~isequal(szsc,szrgb(1:ndsc))
    error('Sizes of rgb and sc inputs must be consistent')
end


[imx, indxx] = mip(sc,2) ;
[imy, indxy] = mip(sc,1) ;
[imz, indxz] = mip(sc,3) ;

eshow(imz,'vdim',vdims(1:2))
eshow(imx,'vdim',[vdims(1) vdims(3)])

rgbz = zeros([ny nx 3]) ;
rgbx = zeros([ny nz 3]) ;

for iy = 1:ny
    for ix = 1:nx
        rgbz(iy,ix,:) = squeeze(rgb(iy,ix,indxz(iy,ix),:)) ;
    end
end

for iy = 1:ny
    for iz = 1:nz
        rgbx(iy,iz,:)=squeeze(rgb(iy,indxx(iy,iz),iz,:)) ;
    end
end

eshow(rgbz,'vdim',vdims(1:2),'isrgb',true)
eshow(rgbx,'vdim',[vdims(1) vdims(3)],'isrgb',true)

return
imdat = zeros(ny+nz, nx+nz) ;
imdat(1:ny, 1:nx) = imz ;
imdat(ny+1:ny+nz , 1:nx) = flipud(rot90(imy)) ;
imdat(1:ny , nx+1:nx+nz) = imx ;

eshow(imdat)


function [im_out, indx] = mip(vol, proj_dim)
% MIP Maximum intensity projection
%  [im_out, indx] = mip(vol, proj_dim)
%
% David Atkinson
% Guy's, King's and St. Thomas' School of Medicine
% %W% , created %G%
%
% See also MIPGUI

[im_out, indx] = max(vol,[],proj_dim) ;

im_out = squeeze(im_out) ;
indx = squeeze(indx) ;

