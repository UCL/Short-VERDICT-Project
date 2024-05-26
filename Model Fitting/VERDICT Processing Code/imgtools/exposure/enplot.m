function enplot
% ENPLOT Plot Bluetooth BLE devices for FD6F Exposure Notification
%
% Addresses change every 10-20 minutes for Exposure Notification.
% Colour of line will change.
% 
% D.Atkinson@ucl.ac.uk
%

hf = figure ;
ha = gca ;

uad = {} ;

while 1 == 1
    list = blelist('services','FD6F','Timeout',10) ;
    if ~isempty(list)
        nl = height(list) ;
        
        dt = datetime('now') ;
        RSSI = list.RSSI ;  % Signal strengths
        
        for ilist = 1: nl
           [~, loc] = ismember(uad, list.Address(ilist)) ;
           if ~any(loc)
               % Newly detected address
               uad = [uad ; list.Address(ilist) ] ;
               iuad = length(uad) ;
               dts = dt ;
               rss = RSSI(ilist) ;
           else
               % existing address
               iuad = find(loc) ;
               dts = [sig(iuad).tm ; dt] ;
               rss = [sig(iuad).rs ; RSSI(ilist) ] ;
           end
           
           sig(iuad).tm = dts ;
           sig(iuad).rs = rss ;
        end
        
        % Plot each unique address as a separate line. Works OK unitl
        % automatic color cycling means that close lines sometimes have the
        % same colour.
        % Ideally "join" addesses that are obvioulsy changes. Criteria
        % might be when new address detected, if it is within +/-20dBm of a 
        % previous address, and that address is no longer visible, then merge.
        
        for iuad = 1:length(uad)
          plot(ha, sig(iuad).tm, sig(iuad).rs), hold on
        end
        ha.YLim = [-105 -50] ;
        grid on
        ylabel('RSSI (dBm)')
        hold off, drawnow    
    end
end
