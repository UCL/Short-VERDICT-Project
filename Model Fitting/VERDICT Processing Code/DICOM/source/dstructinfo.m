function dstructinfo(dstruct)
% DSTRUCTINFO Information about a DICOM structure. Useful for exploring
% data, including multiple series.
%
% Displays information about the structure created from DICOM file(s) when
% using DFPARSE. 
% Selecting (highlighting) one node in each dimension allows a frame 
% to be displayed as an image.
% Checking/unchecking nodes suggests the input string for D2MAT.
%
% David Atkinson
%
% See also D2MAT DFPARSE DSELECTOR

fields = {'SeriesNumber', ...
    'sl', ...
    'InversionTime',...
    'EchoNumbers', ...
    'EffectiveEchoTime', ...
    ... 'itype', ...
    'TemporalPositionIdentifier', ...
    'FlipAngle', ...
    'ImageType', ...
    'DiffusionBValue', ...
    'DiffusionDirectionality', ...
    'DiffGradOrientIdentifier', ...
    } ;

% Corresponding names used in d2mat
d2matnames = {'series', ...
    'slice',...
    'InversionTime', ...
    'echo', ...
    'effTE', ...
    'dyn', ...
    'fa', ...
    'itype', ...
    'bv', ...
    'ddty', ...
    'bdirec', ...
    } ;

xpos = 20 ; % Position of check box tree in figure
ypos = 100 ;
idim = 1 ;  % Dimension counter

% Print values for debugging and set Diffusion Gradient Identifier to 0 if
% empty (otherwise unique will not work as desired later).
hdstr = 'Frame ';
for ifield = 1: length(fields)
    if isfield(dstruct,fields{ifield})
        hdstr = [hdstr d2matnames{ifield} ' '] ;
    end
end
disp(hdstr)

for iframe = 1: length(dstruct) 
    strfr = [num2str(iframe),' '];
    for ifield = 1: length(fields)
        if isfield(dstruct,fields{ifield})
            switch fields{ifield}
                case 'DiffGradOrientIdentifier'
                    if isempty(dstruct(iframe).(fields{ifield}))
                        dstruct(iframe).(fields{ifield}) = 0 ;
                    end
            end
            fval = dstruct(iframe).(fields{ifield}) ; % Field value
            strfr = [strfr num2str(fval) ' '] ;
        end
    end
    disp(strfr)
end

% Find useful text for figure name
figstr = ' ' ;
if isfield(dstruct,'SeriesNumber')
    sns = {dstruct.SeriesNumber} ;
    usns = unique([sns{:}]) ;
    if length(usns) == 1
        figstr = [ figstr, '[' num2str(usns) '] ' ] ;
    else
        figstr = [ figstr, '[', num2str(usns(1)), ' ... ', num2str(usns(end)), '] ' ] ;
    end
end

if isfield(dstruct,'ProtocolName')
    pns = {dstruct.ProtocolName} ;
    upns = unique(pns) ;
    if length(upns) == 1
        figstr = [figstr upns{:}] ;
    else
        figstr = [figstr '(multiple protocols) ' ] ;
    end
end

fig = uifigure('Name',figstr) ;

% Loop through possible fields in input dstruct
for ifield = 1: length(fields)
    if isfield(dstruct,fields{ifield})
        
        fvals = {dstruct.(fields{ifield})} ; % Field values
        if ischar(fvals{1})   % ImageType is a string, with corresponding itype integer
            [uqfvals, ia, ic] = unique(fvals) ;
        else
            [uqfvals, ia, ic] = unique([fvals{:}]) ;
        end
        nuqval = length(ia) ;
        if nuqval > 1
            disp(fields{ifield})

            cbt(idim) = uitree(fig,'checkbox') ;
            cbt(idim).Position = [xpos ypos 150 300] ;
            n1(idim) = uitreenode(cbt(idim)) ;
            n1(idim).Text = fields{ifield} ;
            n1(idim).Tag = 'dimName' ;
            n1(idim).NodeData = d2matnames{ifield};

            neachuqfval = zeros([1 nuqval]) ;
            for iuval = 1:nuqval % loop over each unique value
                inc = find(ic==iuval) ;
                neachuqfval(iuval) = length(inc) ;
                str = uqfvals(iuval) +" ("+ neachuqfval(iuval)+")" ;
                disp(str) 

                n2(iuval) = uitreenode(n1(idim)) ;
                n2(iuval).Text = str ;
                switch fields{ifield} 
                    case 'ImageType'
                        fitvals = [dstruct.itype]; % field itype
                        TF = contains(fvals, uqfvals{iuval} ) ;
                        loc = find(TF==1) ;
                        ND = struct('dimind', fitvals(loc(1)) );
                        ND.locin = loc ;
                    otherwise
                        ND = struct('dimind', uqfvals(iuval) );
                        ND.locin = find([fvals{:}] == uqfvals(iuval)) ;
                end
                % The NodeData stores the locations in dstruct of the
                % entries and the 'dimension indices' (not best word)
                n2(iuval).NodeData = ND ;
                n2(iuval).Tag = 'val' ; % To enable subsequent search to 
                                        % distinguish from top level nodes
            end

            % Expand the trees, check all, select mid values (for plot)
            expand(cbt(idim))
            cbt(idim).SelectedNodes = n2(floor(nuqval/2)) ;
            cbt(idim).CheckedNodes = cbt(idim).Children.Children ;
            cbt(idim).CheckedNodesChangedFcn = @nodechange;
            cbt(idim).SelectionChangedFcn = @nodeselection;

            idim = idim + 1;
            xpos = xpos + 170 ;
        end
    end
end

dd = uidropdown(fig);
dd.Items = {'fp','dv','pv', 'pv_single'} ;
dd.Position = [20 60 100 20] ;
dd.ValueChangedFcn = @nodeselection ;

edt = uieditfield(fig) ;
widthedt = 450 ;
edt.Position = [20 20 widthedt 20] ;

fpos = fig.Position ;
fpos(3) = max([xpos (widthedt+10)]) ;
fig.Position = fpos ;



    function nodeselection(src, event)
        % selection means highlight 
        hh = src.Parent.Children ; % handles in fig
        cbts = findobj(hh,'Type','uicheckboxtree') ;
        locrunning = [1:length(dstruct)];
        for icbt = length(cbts):-1:1
            sn = cbts(icbt).SelectedNodes ;
            if length(sn) ~= 1 
                disp('Needs exactly one selection to display image')
            else
                if strcmp(sn.Tag, 'dimName')
                    % ignore if top is highlighted
                else
                    locrunning = intersect(locrunning, sn.NodeData.locin) ;
                end
            end
        end

        if length(locrunning) ~= 1
            disp([num2str(length(locrunning)), ' images (must be exactly 1)']) 
        else
            X = dicomread(dstruct(locrunning).Filename,'frames',dstruct(locrunning).Frame) ;
            switch dd.Value
                case 'pv_single'
                    X = single(X) ;
                case 'dv'
                    X = single(X)*dstruct(locrunning).RescaleSlope + ...
                        dstruct(locrunning).RescaleIntercept ;
                case 'fp'
                    X = (single(X)*dstruct(locrunning).RescaleSlope + ...
                                          dstruct(locrunning).RescaleIntercept) / ...
                           (dstruct(locrunning).RescaleSlope * dstruct(locrunning).Private_2005_100e) ;
            end
            
            guifig = src.Parent ;
            UD = guifig.UserData ;
            if ~isfield(UD,'axselect') || ~ishandle(UD.axselect)
                UD.hf = figure ;
                UD.axselect = gca ;
                guifig.UserData = UD ;
            end

            him = imshow(X,[],'Parent',UD.axselect);
            imdisplayrange(UD.hf,him)
            disp(['Figure ',num2str(UD.hf.Number),' updated.'])
        end

    end

    function nodechange(src,~)
        % D2MAT has a cell array of input dimensions and a cell array of
        % restrictions. 
        % d2fulldim are dimensions without restrictions
        % d2middim  are restricted dimensions with more than one value
        % d2sgldim  are dimensions with one value allowed. Here placed at
        % the end of the d2mat call to 'loose' the singleton dimension.
        d2fulldim = {} ; % used for building arguments to d2mat
        d2middim = {} ;
        d2sgldim = {} ;
        d2matrestrict = {} ;
        hh = src.Parent.Children ; % handles in fig

        cbts = findobj(hh,'Type','uicheckboxtree') ;
        % When all nodes in a tree are checked, the parent node is also 
        % included in list! - complicates processing!

        for icbt = length(cbts):-1:1
           cn = cbts(icbt).CheckedNodes ; % will include top node if it is checked
           
           dcn = findobj(cn,'flat','Tag','dimName') ; % top node checked
           if ~isempty(dcn)
               % all are selected (top node is checked)
               d2fulldim{end+1} = dcn.NodeData ;
           else
               % the top node is not checked
               % see how many nodes below are checked
               vcn = findobj(cn,'flat','Tag','val') ; % checked nodes exc top
               if length(vcn) == 1
                   % just one, make this a singleton on the end of d2mat
                   dimname = cbts(icbt).Children.NodeData ;
                   tND = vcn.NodeData ;
                   dimval = tND.dimind ;
                   d2sgldim{end+1} = dimname ;
                   d2matrestrict{end+1} = dimname ;
                   d2matrestrict{end+1} = num2str(dimval) ;
               elseif length(vcn) > 1
                   % a mid dimension
                   dimname = cbts(icbt).Children.NodeData ;
                   tND = [vcn.NodeData] ;
                   dimvals = [tND.dimind] ;
                   d2middim{end+1} = dimname ;
                   d2matrestrict{end+1} = dimname ;
                   d2matrestrict{end+1} = ['[ ',num2str(dimvals), ' ]'] ;
               end
           end
        end
        
        % Compute a string that is the d2mat input arguments
        dstr = []; drstr  = [] ;
        ap = char("'") ; % single quote
        for id = 1:length(d2fulldim)
            dstr = [dstr, [ap,d2fulldim{id}, ap,',']];
        end
        for id = 1:length(d2middim)
            dstr = [dstr, [ap,d2middim{id}, ap,',']];
        end
        for id = 1:length(d2sgldim)
            dstr = [dstr, [ap,d2sgldim{id}, ap,',']];
        end

        for id = 1:2:length(d2matrestrict)
            drstr = [drstr, ap,d2matrestrict{id}, ap,',',d2matrestrict{id+1},',']; 
        end

        if ~isempty(d2matrestrict)
            d2matstr = [ '[v,m,l] = d2mat(dstruct,{',dstr(1:end-1),'} ', drstr(1:end-1) ,', ''op'',''',dd.Value , ''') ;'] ;
        else
            d2matstr = [ '[v,m,l] = d2mat(dstruct,{',dstr(1:end-1),'}, ''op'',''',dd.Value , ''') ;'] ;
        end

        % Display the string in edit box for copy-paste
        edt.Value = d2matstr ;

    end
end









