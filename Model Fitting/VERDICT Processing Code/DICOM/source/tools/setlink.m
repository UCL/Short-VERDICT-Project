function setlink(src,event,ltype,askfig)
% SETLINK Sets the links for axes, CLim and slices (z link)
%
% setlink(src, event, ltype) 
%
%  Called by sviewer context menu
%
% David Atkinson
%
% See also sviewer

% Get all the open sviewer figures
FKEY = 'sviewer_fig' ;

if nargin < 4
   askfig = true ;
end

hfigs = findall(groot,'Tag',FKEY) ;
UD = get(groot,'UserData');
if isempty(UD) || ~isfield(UD,'glinks') 
    UD.glinks = [] ;  % links for CLim stored in groot (for zlink, code uses zlhfigs)
end

for ifig = 1:length(hfigs)
    hfthis = hfigs(ifig) ;
    ax = findobj(hfthis,'Type','axes') ;
    fnum = hfthis.Number ;
    fnum2ax(fnum) = ax ;
    fnum2hf(fnum) = hfthis ;
    allfnum(ifig) = fnum ;
    disp(['Fig ',num2str(fnum),' ',hfthis.Name]) ;
end

switch ltype
    case 'all'
        askfig = false ;
        setlink(src,event,'axes', askfig)
        setlink(src,event,'zlink', askfig)

    case 'axes'
        if askfig
            fignums = input('Enter figure numbers to link axes, e.g. [1 3]. ') ;
        else
            fignums = allfnum ;
        end
        haxs = axfromfignum(fignums, fnum2ax) ;
        linkaxes(haxs)
    case 'axes unlink'
        fignums = input('Enter figure numbers to unlink axes, e.g. [1 3]. ') ;
        haxs = axfromfignum(fignums, fnum2ax) ;
        linkaxes(haxs,'off')
    case 'CLim'
        fignums = input('Enter figure numbers to link CLim, e.g. [1 3]. ') ;
        haxs = axfromfignum(fignums, fnum2ax) ;
        hlink = linkprop(haxs,'CLim') ;
        UD.glinks = cat(2,UD.glinks,hlink);
        set(groot,'UserData',UD) 
    case 'CLim unlink'
        glinks = UD.glinks ;
        if ~isempty(glinks) 
            for ilink = 1:length(glinks)
                if isvalid(glinks(ilink))
                    figv = [] ;
                    for itarget = 1: length(glinks(ilink).Targets)
                        figv = cat(2,figv,glinks(ilink).Targets(itarget).Parent.Number);
                    end
                else
                    figv = [] ;
                end
                disp(['Link number ',num2str(ilink),' (links Figs: ',num2str(figv),')'])
            end
            linknum = input('Enter link number to delete: ');
            delete(glinks(linknum))
        end
    case 'zlink'
        allfigs = findall(groot,'Tag','sviewer_fig') ;
        for ifig = 1:length(allfigs)  % set all figure iszlinked to false
            UDf = allfigs(ifig).UserData ;
            UDf.iszlinked = false ;
            allfigs(ifig).UserData = UDf;
        end
        if askfig
            fignums = input('Enter figure numbers to Zlink, e.g. [1 3]. ') ;
        else
            fignums = allfnum ;
        end
        
        hfs = fnum2hf(fignums) ;
        for ifig = 1:length(hfs)
            UDf = hfs(ifig).UserData ;
            UDf.iszlinked = true ;
            hfs(ifig).UserData = UDf ;
        end
        UDg = get(groot,'UserData') ;
        UDg.zlhfigs = hfs ;
        set(groot,'UserData',UDg)
    otherwise
        warning(['Unknown link type: ',ltype])
end
end

function haxs = axfromfignum(fignums, fnum2ax)
haxs = [] ;
for ifig = 1:length(fignums)
    thisax = fnum2ax(fignums(ifig)) ;
    haxs = cat(1,haxs, thisax) ;
end
end
