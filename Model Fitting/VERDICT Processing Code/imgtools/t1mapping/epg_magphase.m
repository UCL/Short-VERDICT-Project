function epg_magphase(arr, varargin)
% EPG_MAGPHASE Plots magnitude and phase fro extended phase graphs
% Based on magphase from Brian Hargreaves
%

ernst = [] ;
for ip = 1:2:length(varargin)
    switch varargin{ip}
        case 'Ernst'
            ernst = varargin{ip+1} ;
        case 'title'
            titlestr = varargin{ip+1} ;
        otherwise
            warning(['Unknown parameter: ',varargin{ip}])
    end
end

x = 1:length(arr);

mg = abs(arr);
ph = angle(arr);

subplot(2,1,1);
plot(x,mg);
ylabel('Magnitude');
if ~isempty(titlestr)
    title(titlestr)
end
grid on
a = axis;
% axis([a(1) a(2) 0 ceil(10*a(4))/10])
axis([a(1) a(2) 0 0.1])
if ~isempty(ernst)
   hold on
   plot([1 x(end)],[ernst ernst],'LineStyle',':','LineWidth',1)
end


subplot(2,1,2);
plot(x,ph/pi);
ylabel('Phase/\pi');
a = axis;
axis([a(1) a(2) -1 1]);
grid on;




