function reconreveal(MR, varargin)
% REONREVEAL
%
%  reconreveal(MR)
%  reconreveal(MR, 'output', outstr, 'comment', cmnt)  
%    str letters encode output 'S' data size, 'E' encoding paraemters
%    e.g.  reconreveal(MR,'output,'SE', 'comment', 'after ReadData')

outencp = false ;
outsz = false;
comment = '';

for ipv = 1:2:length(varargin)
   val = varargin{ipv+1} ;
   switch varargin{ipv}
       case 'output'
           if contains(val,'S')
               outsz = true;
           end
           if contains(val,'E')
               outencp = true ;
           end
       case 'comment'
           comment = val ;
       otherwise
           warning('unknown parameter')
   end
end

if iscell(MR.Data)
    d = MR.Data{1} ; 
else
    d = MR.Data ;
end

str = sprintf('Data: %3d %3d %3d %2d %2d %2d %2d %2d %2d %2d %2d', ...
    size(d,1), size(d,2), size(d,3), size(d,4), size(d,5), size(d,6), ...
    size(d,7), size(d,8), size(d,9), size(d,10), size(d,11)) ;

if outsz == true
    disp([comment,' ',str])
else
    disp(comment)
end

KxRange = MR.Parameter.Encoding.KxRange ;
KyRange = MR.Parameter.Encoding.KyRange ;
XRes    = MR.Parameter.Encoding.XRes ;
KxOversampling = MR.Parameter.Encoding.KxOversampling ;
YRes    = MR.Parameter.Encoding.YRes ;
KyOversampling = MR.Parameter.Encoding.KyOversampling ;
XRange = MR.Parameter.Encoding.XRange ;
YRange = MR.Parameter.Encoding.YRange ;
XReconRes = MR.Parameter.Encoding.XReconRes ;
YReconRes = MR.Parameter.Encoding.YReconRes ;

SENSEFactor = MR.Parameter.Scan.SENSEFactor ;
SENSEExtraOversampling = MR.Parameter.Scan.SENSEExtraOversampling ;


if outencp == true
    disp( sprintf('KxRange [ %d, %d ] (%d), KyRange [ %d, %d ] (%d)', ...
        KxRange, (KxRange(2)-KxRange(1)+1), KyRange, (KyRange(2)-KyRange(1)+1)))
    disp( sprintf('XRes %d * KxOversampling %d = (%d). YRes %d * KyOversampling %d = (%d)', ...
        XRes, KxOversampling, XRes*KxOversampling, ...
        YRes, KyOversampling, YRes*KyOversampling ) )
    
    if (XRange(2)-XRange(1) + 1) ~= XRes
        xw = '!! XRange ~= XRes';
    else
        xw = ' ';
    end
    if (YRange(2)-YRange(1) + 1) ~= YRes
        yw = '!! YRange ~= YRes';
    else
        yw = ' ' ;
    end
    
    disp( sprintf('XRange [ %d, %d ] (%d) %s, YRange [%d, %d ] (%d) %s' , ...
        XRange, (XRange(2)-XRange(1)+1), xw, YRange, (YRange(2)-YRange(1)+1), yw))
    
    disp( sprintf('XReconRes %d, YReconRes %d', XReconRes, YReconRes) )
    disp( sprintf('SENSEFactor [%f, %f, %f ]', SENSEFactor ))
    if ~isempty(SENSEExtraOversampling)
      disp( sprintf('SENSEExtraOversampling %f, %f, %f', SENSEExtraOversampling) )
    end
    disp(' ')
end
