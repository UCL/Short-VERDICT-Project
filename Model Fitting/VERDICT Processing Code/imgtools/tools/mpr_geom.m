function geom_mpr = mpr_geom(g_in)
% MPR_GEOM Multi-plane Reformat Geometry
%
% geom_mpr = mpr_geom(geom_in)
% geom_mpr = mpr_geom(matp_in)
% 
%  Input can be geom struct, or a structure that includes a 
%   field which is itself a geom struct, for example the matp structure 
%   returned by d2mat
%
% Currently only supports a transverse reformat and only tested with a
% coronal input volume. (Written for MPR of 3D MERGE COR data).
% XData and YData not correctly set (will not work with roianal)
%
% D.Atkinson@ucl.ac.uk
%
% See also D2MAT
%

ori_mpr = 'trampr' ;

if isfield(g_in,'geom')
    geom_in = g_in.geom ;
else
    geom_in = g_in ;
end

[slsep, rdc, cdc, sdc] = geom_check(geom_in) ;

sagnorm = cross([0 1 0],[0 0 -1]) ; % LPH coordinate
cornorm = cross([1 0 0],[0 0 -1]) ;
tranorm = cross([1 0 0],[0 1  0]) ;

sagproj = dot(sagnorm, sdc) ; 
corproj = dot(cornorm, sdc) ;
traproj = dot(tranorm, sdc) ;

[~, iproj] = max([sagproj corproj traproj]);
volstr = {'sagvol', 'corvol', 'travol'} ;
ori_vol = volstr{iproj} ;
disp(['Input volume is: ',ori_vol])

nslice_in = length(geom_in) ;
gin1 = geom_in(1) ;

switch ori_vol
    case 'corvol'
        aL = rdc ;
        aP = sdc ;
        aH = -cdc ;
        
        psvL = gin1.PixelSpacing_HW(1) ;
        psvP = slsep ;
        psvH = gin1.PixelSpacing_HW(2) ;
        
        fovL = gin1.Width * psvL ;
        fovP = nslice_in * slsep ;
        fovH = gin1.Height * psvH ;
        
        vcorner = gin1.IPP + -0.5*psvL*rdc -0.5*psvH*cdc -0.5*psvP*sdc
%        vcent = vcorner + fovH/2*psvH*cdc + fovL/2*psvL*rdc + ...
%            slsep*nslice_in/2*psvP*sdc 
        vcent = vcorner + fovH/2*cdc + fovL/2*rdc + ...
            slsep*nslice_in/2*sdc 

    otherwise
        disp(['Only COR input vols not tested.'])
end

switch ori_mpr
    case 'trampr'
        IOP = [aL ; aP]' ;
        
        psmP = psvP ;
        psmL = psvL ;
        psmH = psvH ;
        
        %make in-plane equal
        inplane = min([psmP psmL]) ;
        
        PixelSpacing_HW(1) = inplane ;
        PixelSpacing_HW(2) = inplane ;
        SliceThickness = psmH ;
        
        Width = ceil( fovL / PixelSpacing_HW(2) ) ;
        Height = ceil( fovP / PixelSpacing_HW(1) ) ;
        
        %make Width and Height equal, update fov
        hw = max([Width Height]) ;
        Width = hw ;
        Height = hw ;
        fovL = Width*PixelSpacing_HW(2) ;
        fovP = Height* PixelSpacing_HW(1) ;
        
        nslice_out = ceil( fovH / SliceThickness) ;
        nsl = nslice_out ;
        
        geom_mpr = struct('IPP',num2cell(zeros([nsl 3]),2), ...
           'IOP',num2cell(repmat(IOP,[nsl 1]),2), ...
           'Height',num2cell(repmat(Height,[nsl 1])), ...
           'Width',num2cell(repmat(Width,[nsl 1])), ...
           'PixelSpacing_HW',num2cell(repmat(PixelSpacing_HW,[nsl 1]),2), ...
           'SliceThickness',num2cell(repmat(SliceThickness,[nsl 1])), ...
           'XData',num2cell(zeros([nsl 2]),2), ...
           'YData',num2cell(zeros([nsl 2]),2), ...
           'source',{'MPR'}) ;
        sledge = vcent + (SliceThickness*nslice_out)/2*(-aH) ;
        for islice =1:nslice_out
            slcent = sledge + (islice-0.5)*SliceThickness * aH ;
            %IPP = slcent -aP*PixelSpacing_HW(1)*fovP/2 + PixelSpacing_HW(1)/2*aP + ...
            %    -aL*PixelSpacing_HW(2)*fovL/2 + 0.5*PixelSpacing_HW(2)*aL;
            IPP = slcent -aP*fovP/2 + PixelSpacing_HW(1)/2*aP + ...
                    -aL*fovL/2 + 0.5*PixelSpacing_HW(2)*aL;
            
            geom_mpr(islice).IPP = IPP' ;
          
        end
    otherwise
        disp(['Only trampr implemented'])
end



