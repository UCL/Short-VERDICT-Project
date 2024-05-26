function copyPhilipsPrivate(infiles, outfolder)
% COPYPHILIPSPPRIVATE Copy Philips Private Fields (for scanner re-import)
%
%
% David Atkinson, University College London
%

dictJoined = joindictprivate ;
dicomdict('set',dictJoined)

writeArgs = {'WritePrivate', true, ...
        'MultiframeSingleFile', false, 'VR', 'explicit'} ;

for ifile = 1:length(infiles)
    this_file = infiles{ifile} ;
    if ~isdicom(this_file)
        error(['Input file must be DICOM: ',this_file])
    end

    dinfo = dicominfo(this_file) ;
    [X, cmap] = dicomread(this_file) ;

    fnstemext = ['test-',num2str(ifile,'%03d'), '.dcm'] ;
    outfile = fullfile(outfolder,fnstemext) ;

    if isempty(cmap)
        status = dicomwrite(X,outfile,dinfo, writeArgs{:}) ;
    else
        status = dicomwrite(X,cmap, outfile,dinfo, writeArgs{:}) ;
    end

end


dicomdict('factory')
disp(['Written to files in:', outfolder])