function lab = parreadall(fn)
% PARREADALL Read a V4.2 PAR file that labels the REC data.
%
% Note needs updating to replace strread with textscan - see paranon.m for
% examples
%
% David Atkinson  D.Atkinson@ucl.ac.uk
%


if nargin == 0
  [df,dp] = defuigetfile('*.*','Select PAR file') ;  % could be par or PAR
  fn = fullfile(dp,df) ;
end

[fid, message] = fopen(fn,'rt') ;
if fid < 2
  error(message)
end

parfn = fn ;
[pstr,nm,ext]=  fileparts(parfn) ;
switch ext
 case '.PAR'
  recfn = [fullfile(pstr,nm),'.REC'] ;
 case '.par'
  recfn = [fullfile(pstr,nm),'.rec'] ;
end

lab.RecFileName = recfn ;
lab.ParFileName = parfn ;

fseek(fid,0,1) ; % eof
len = ftell(fid) ;
fseek(fid,0,-1) ; % bof

% assume that lines starting
% # are comments
% . are general info (note format different from RAW and CPX .data
% files)
% ' ', if not blank, are lines of image information
%


curr_offset = 0 ; % current offset (a running total)
nidx = 0 ;

while ftell(fid) < len
  
  ln = fgetl(fid) ;
  if length(ln) == 0
    ln = 'blank';
  end
  
  switch ln(1)
   case '#'
    k = strfind(ln,'Dataset name') ;
    if length(k) > 0
      lab.dataset_name = ln(17:end) ;
    end
    k = strfind(ln,'CLINICAL') ;
    if length(k) > 0
      lab.ver = ln(62:65) ;
    end
   case '.'
      [precolon, postcolon] = strread(ln,'.%[^:]:%s','whitespace', ...
				      '\b\t');
      params = deblank(precolon{1}) ;
      
      switch params
          case 'Patient name'
              lab.PatientName = postcolon{1} ;
          case 'Examination name'
              lab.ExaminationName = postcolon{1} ;
          case 'Protocol name'
              lab.ProtocolName = postcolon{1} ;
          case 'Acquisition nr'
              lab.AcqNr = str2num(postcolon{1}) ;
          case 'Reconstruction nr'
              lab.RecNr = str2num(postcolon{1}) ;
          case 'FOV (ap,fh,rl) [mm]'
              [Fap, Ffh, Frl] = strread(postcolon{1},'%f%f%f') ;
              lab.FOV_lph = [Frl Fap Ffh];
          case 'Angulation midslice(ap,fh,rl)[degr]'
              [Aap, Afh, Arl] = strread(postcolon{1},'%f%f%f') ;
              lab.AngulationMidslice_lph = [Arl Aap Afh] ;
          case 'Off Centre midslice(ap,fh,rl) [mm]'
              [Oap, Ofh, Orl] = strread(postcolon{1},'%f%f%f') ;
              lab.OffcentreMidslice_lph = [Orl Oap Ofh] ;
              %[nx ny] = strread(postcolon{1},'%u%u') ;
      end
      
   case {' ','1','2','3','4','5','6','7','8','9'} % for no. > 9, no space at start
    
   % [sl, ec, dyn, card, imtyp, scnseq, recindx, ri, rs, ss, wc, ww, ...
   %  imang(AP), imang(FH), imang(RL), imoff(AP), imoff(FH), imoff(RL),...
   %  imdispori, sliori, fmri, endds, pixel_spacing(XC), pixel_spacing(YC), ...
   %  ect, dynt, tt, bfac, flip] = strread(ln,'%7u%3f%2u%6f%4u%7f') ;
   
    nidx = nidx + 1 ;
    
    recinf = strread(ln,'%f') ;
    
    im.loca = recinf(1) ;
    im.echo = recinf(2) ;
    im.dyn  = recinf(3) ;
    im.card = recinf(4) ;
  
    im.IndexRec = recinf(7) ;
    bitspv = recinf(8) ;
	im.bypv = bitspv/8 ;
    nx = recinf(10) ;
    ny = recinf(11) ;
    im.sz = [ny nx] ;
    im.RescaleIntercept = recinf(12) ;
    im.RescaleSlope = recinf(13) ;
    im.ScaleSlope = recinf(14) ;
    im.WindowCentre = recinf(15) ;
    im.WindowWidth = recinf(16) ;
    im.ImageAngulation_lph = [recinf(19) recinf(17) recinf(18)];
    im.Offcentre_lph = [recinf(22) recinf(20) recinf(21)];
    im.SliceThickness = recinf(23) ;
    im.SliceGap = recinf(24) ;
    im.ImageDisplayOrientation = recinf(25) ;
    oristr = ['TRA' ; 'SAG' ; 'COR'] ;
    im.SliceOrientation = oristr(recinf(26),:) ;
    im.PixelSpacing_hw = [recinf(30) recinf(29)] ;
    im.EchoTime = recinf(31) ;
    im.DynScanBeginTime = recinf(32) ;
    im.TriggerTime = recinf(33) ;
    im.DiffusionBFactor = recinf(34) ;
    im.NSA = recinf(35) ;
    im.FlipAngle = recinf(36) ;
    
    im.TurboFactor = recinf(40) ;
    im.InvDelay = recinf(41) ;
    im.DiffusionBValueIdentifier = recinf(42) ;
    im.DiffGradOrientIdentifier = recinf(43) ;
    im.DiffusionGradientOrientation(2) = recinf(46) ;
    im.DiffusionGradientOrientation(3) = recinf(47) ;
    im.DiffusionGradientOrientation(1) = recinf(48) ;
    
    
    im.RecImSizeBytes = im.sz(1)*im.sz(2)*im.bypv ;
    im.RecOffsetBytes = im.RecImSizeBytes*im.IndexRec ;
    
    par{nidx} = im ;
    
   case 'b'
    %ignore blank line
   otherwise
    error(['Unknown start to line : ',ln])
  end
end

spar = [par{:}] ;
% Check all images were same size, otherwise error in offset assumptions
rsb = [spar.RecImSizeBytes] ;
uq = unique(rsb) ;
if length(uq) > 1
    warning(['More than one image size in REC file - offset calc needs updating'])
end
disp(['REC file should have size ',num2str(sum(rsb)),' bytes.'])
lab.RecFileSize = sum(rsb) ;


lab.par = spar ;

fclose(fid) ;
disp([' Read from file ',fn,])







