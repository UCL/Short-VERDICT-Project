function varargout = XXread(file_name, varargin)
% Read Philips XX files (DICOM Raw Data Storage containing exam parameters)
%  Useful for finding exam and info-page parameters that were present 
%  during scanning. 
%
%  XXread  -  calls UI for input file name, output to temp file in editor
%  XXread(filename) - read XX filename, output to temp file in editor
%  XXread(filename, Name, Value, ...)
%  xxinfo = XXread... output to a structure composed of the parameters 
%                    decoded from the XX file, no output file by default
%
%  Name, Value pairs
%   'fout', {true}  false  Output file produced
%   'outFileNaming': {'tempname'}, 'XXstem', Naming of output file, either a
%   temporary file or the input XX filename is used as stem for a default
%   'outFileInEditor' {true}, false
%   'verbose' true or {false}
%   'suppressNotXXWarning' {false}
%
%
%  Note all XX parameters are output and may have default values that 
%  are meaningless for the specific scan. It is not clear when the 
%  parameter values are taken (before or after scan?).
%
%
% The XX file is a DICOM format file output from Philips MR scanners. It is
% not always transmitted through PACs systems. There is one XX file per
% series and often additional XX file related to the whole ExamCard.
%
% These files are 'RAW' with SOPClassUID '1.2.840.10008.5.1.4.1.1.66' 
%
% THIS SOFTWARE IS PROVIDED  "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
% POSSIBILITY OF SUCH DAMAGE.
%
% Requires dicomread in Image Processing Toolbox.
%
% COILSTATE and HARDWARE_CONFIG present in R5.1.7 XX files are not 
% currently extracted.
%
% See https://github.com/malaterre/GDCM/blob/master/Applications/Cxx/gdcmdump.cxx
% Process SDS data
%
% Partial decode of the ExamCard XX file in XXsummary. For possible clues, see:
% https://github.com/IBIC/ibicUtils/blob/master/ExamCard.py
% https://irc.cchmc.org/software/libexamcard/doc/examcard2xml_8c-example.html
%
% David Atkinson  D.Atkinson@ucl.ac.uk
%
% See also DICOMINFO XXSUMMARY xxdiff

% From https://github.com/malaterre/GDCM/blob/master/Applications/Cxx/gdcmdump.cxx
%
% static void ProcessSDSData( std::istream & is )
% {
%   // havent been able to figure out what was the begin meant for
%   is.seekg( 0x20 - 8 );
%   uint32_t version = 0;
%   is.read( (char*)&version, sizeof(version) );
%   assert( version == 8 );
%   uint32_t numel = 0;
%   is.read( (char*)&numel, sizeof(numel) );
%   for( uint32_t el = 0; el < numel; ++el )
%     {
%     PDFElement pdfel;
%     assert( sizeof(pdfel) == 50 );
%     is.read( (char*)&pdfel, 50 );
%     if( *pdfel.getname() )
%       {
%       printbinary( is, pdfel );
%       std::cout << std::endl;
%       }
%     }
%
% }

verb = false; % VERBOSE, set true for debugging/screen output.
outFileInEditor = true ;
outFileNaming = 'tempname' ;
suppressNotXXWarning = false ;

if nargout > 0
    sout = true ; % structure output
    fout = false ;
else
    sout = false ;
    fout = true ;
end


if length(varargin)>1
    for ipv = 1:2:length(varargin)
        val = varargin{ipv+1} ;
        switch varargin{ipv}
            case 'verbose'
                verb = val ;
            case 'fout'
                fout = val ;
            case 'outFileInEditor'
                outFileInEditor = val ;
            case 'outFileNaming'
                outFileNaming = val ;
            case 'suppressNotXXWarning'
                suppressNotXXWarning = val ;
            otherwise
                error(['Unrecognised input', varargin{ipv}])
        end
    end
end




if nargin >=1
    if iscell(file_name)
        file_name = file_name{1} ;
    end
end

% Determine XX file name. Use pref_uigetfile if present, otherwise native 
% MATLAB uigetfile. Quit if user cancels UI.
if nargin <1 || ~exist(file_name,'file')
    disp('Select XX file')
    if ~exist('pref_uigetfile')
        [fxx,pxx,~] = uigetfile('*.*','Enter the XX file') ;
        if isequal(fxx,0)
            varargout = {} ;
            return
        else
            file_name = fullfile(pxx, fxx) ;
        end
    else
        file_name = pref_uigetfile('XXread','filename') ;
        if isempty(file_name)
            varargout = {} ;
            return
        end
    end
    
end
if sout ; xxinfo.Filename = file_name; end

% Output file and name

[p,n,~] = fileparts(file_name) ;
deffn = fullfile(p,[n,'.txt']) ; % default output file name

if fout 
    switch outFileNaming
        case 'tempname'
            ffnout = tempname ;
            [pnout, fnout] = fileparts(ffnout) ;
        case 'XXstem'
            [fnout,pnout] = uiputfile('*.txt','Filename for output',deffn) ;
            if isequal(fnout,0) || isequal(pnout,0)
                warning('No output text file selected.')
                fout = false ;
            end
        otherwise
            error('Unknown outFileNaming value')
    end
end

if fout
    fid = fopen(fullfile(pnout,fnout), 'wt+') ;
end

if ~fout && outFileInEditor
    warning('Cannot open output file in editor as none will be created.')
end

% Reset Dicom dictionary to default so as to preserve the Private tag names
% which could be overwritten if user has a modified dicom dictionary.
% Read Private tags using dicominfo
% Set dicom dictionary back 

dict = dicomdict('get') ;
dicomdict('factory');

% ==== Changed to true!!!!!
dinfo = dicominfo(file_name, UseDictionaryVR = true ) ;
dicomdict('set',dict) ; 

if isfield(dinfo,'SeriesNumber'), xxinfo.SeriesNumber = dinfo.SeriesNumber; end
if isfield(dinfo,'SeriesDate'),   xxinfo.SeriesDate = dinfo.SeriesDate; end
if isfield(dinfo,'SeriesTime'),   xxinfo.SeriesTime = dinfo.SeriesTime; end
if isfield(dinfo,'ProtocolName'), xxinfo.ProtocolName = dinfo.ProtocolName; end


if isfield(dinfo,'Private_2005_1132')  % (2005,1132) SQ MRBlobDataObjectArray 1
    currpf = dinfo.Private_2005_1132 ;
    
    fnb = fieldnames(currpf);
    for iitem = 1:length(fnb)
        crtyp = currpf.(fnb{iitem}).Private_2005_1139 ;  % (2005,1139) MRTypeName
        crtit = currpf.(fnb{iitem}).Private_2005_1137 ;  % (2005,1137) MRBlobName
        
        crabs = currpf.(fnb{iitem}).Private_2005_1143 ; 

        % (2005,1144)	OW	MRBlobData	1
        % (2005,1137)	PN	MRBlobName	1  (e.g [PDF_CONTROL_PREP_PARS] )
        % (2005,1138)	PN	MRApplicationName	1  (empty?)
        % (2005,1143)	SL	MRActualBlobSize	1
        % (2005,1140)	PN	MRVersionStr	1  (empty in XX?)
        % (2005,1141)	PN	MRCommentStr	1  (empty in XX?)
        % (2005,1147)	CS	MRBlobFlag	1

        if isstruct(crtyp) % old style?
            crtyp = crtyp.FamilyName ;
            crtit = crtit.FamilyName ;
        end
        if strcmp(crtit,'COILSTATE'), continue; end % skip this item if it is the COILSTATE info - not yet readable
        if strcmp(crtit,'HARDWARE_CONFIG'), continue; end % skip this item if it is the HARDWARE_CONFIG info  - not yet readable
        if strcmp(crtit,'PDF_PRESCAN_COIL_PARS'), continue, end
        if strcmp(crtit,'PDF_USED_PRESCAN_RESPONSES'), continue, end

        if isa(crtyp,'uint8') % A Dicom transfer issue has probably 
                              % switched type to uint8
            crtyp = char(crtyp') ;
            crtit = char(crtit') ;
        end
        if strcmp(crtit,'ExamCardBlob')
            disp('ExamCardBlob')
            continue
        end
        
        switch crtyp
            case 'BINARY'
                disp(['BINARY - not decoded: ',fnb{iitem}])
            case 'ASCII'
                disp(['ASCII - not decoded: ',fnb{iitem}])
            otherwise
                if verb; disp(['type: ',crtyp]); end
                
                cs = currpf.(fnb{iitem}).Private_2005_1144 ;  % (2005,1144)	MRBlobData
                if isa(cs,'uint8')
                    cs = typecast(cs,'uint16') ;
                end
                if isa(cs,'int8')
                    warning('typecast suggested by online EXCEED code')
                    cs = typecast(cs,'uint16') ;
                end
                n = typecast(uint16(cs(15:16)),'uint32') ;
                % str = [currpf.(fnb{iitem}).Private_2005_1137,' n = ',num2str(n)] ;
                str = [crtit,' n = ',num2str(n)] ;
                if verb ; disp(str) ; end
                if fout ; fprintf(fid,'%s\n',str) ; end
                i8all = typecast(uint16(cs),'uint8') ;
                
                pos = 32 ; % bytes
                
                for ient = 1:n
                    i8 = i8all(pos+1:pos+50) ;
                    loc = find(i8==0) ;
                    str = char(i8(1:loc(1)-1))'; % DA inserted the -1 to prevent 0 in str
                    % disp([str,' ',num2str(i8(loc(1)+1:end)')])

                    typ = typecast(uint8(i8(35:38)),'uint32') ;
                    numelems = typecast(uint8(i8(39:42)),'uint32') ;
                    offset   = typecast(uint8(i8(47:50)),'uint32') ;
                    jump = pos+50+offset-4 ;
                    
                    if verb ; disp([str,' # ',num2str(numelems)]) ; end
                    if fout ; fprintf(fid,'%s\n',[str,' # ',num2str(numelems)]); end
                    if sout; fieldnm = str ; end
                    
                    switch typ
                        case 0
                            typs = 'single' ;
                            vals = typecast(uint8(i8all(jump+1:jump+numelems*4)),typs);
                            nv = length(vals) ;
                            if verb ; disp(num2str(vals(1:min(30,nv))')); end
                            if fout ; fprintf(fid,'%s\n',num2str(vals')); end
                            if sout; fieldval = vals ; end
                            
                        case 1
                            typs = 'int32' ;
                            vals = typecast(uint8(i8all(jump+1:jump+numelems*4)),typs);
                            nv = length(vals) ;
                            if verb; disp(int2str(vals(1:min(30,nv))')) ; end
                            if fout; fprintf(fid,'%s\n',int2str(vals')); end
                            if sout; fieldval = vals ;end
                            
                        case 2
                            typs = 'STR80' ;
                            vals81 = typecast(uint8(i8all(jump+1:jump+numelems*81)),'uint8');
                            vals = []; fieldval = [] ;
                            for istr = 1:numelems
                                instrp = (istr-1)*81 ;
                                valsstr = char(vals81(instrp+1:instrp+80))' ;
                                locend = find(int8(valsstr)==0) ;
                                valsstr = valsstr(1:locend(1)-1) ;
                                if verb; disp(valsstr); end
                                if fout; fprintf(fid,'%s\n',valsstr); end
                                if sout; fieldval{istr} = valsstr ; end
                            end
                            
                        case 4
                            typs = 'uint32' ;
                            vals = typecast(uint8(i8all(jump+1:jump+numelems*4)),typs);
                            nv = length(vals) ;
                            if verb; disp(num2str(vals(1:min(30,nv))')) ; end
                            if fout; fprintf(fid,'%s\n',num2str(vals')); end
                            if sout ; fieldval = vals ; end 
                        otherwise
                            disp(['Unknown type: ',typ])
                    end
                    if sout
                        xxinfo.(fieldnm) = fieldval ;
                    end
                    %disp([str,' # ',num2str(numelems),' type: ',num2str(typs),' offset: ',num2str(offset)])
                    %nv = length(vals) ;
                    %disp(vals(1:min(30,nv))')
                    pos = pos + 50 ;
                end
                if verb; disp(' '); end
                if fout; fprintf(fid,'%s\n',' '); end
        end
        
    end
else
    if ~suppressNotXXWarning
        warning('No Private_2005_1132 found (possibly removed on anonymisation or not an XX file)')
    end
end

if sout
    varargout{1} = xxinfo ;
end

if fout
    fclose(fid);
    disp(['Data written to ',fullfile(pnout,fnout)])

    if outFileInEditor
        edit(fullfile(pnout,fnout))
    end
end

