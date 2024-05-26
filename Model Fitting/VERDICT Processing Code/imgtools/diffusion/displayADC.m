function displayADC(ADC, ADCtop)
% Displays ADCs as a montage with a numbered colorbar
%
% displayADC(ADC, ADCtop)
% displayADC(ADC)  top will default to 1700
% Assumes units where ADC is around 800 in brain
%
%
% David Atkinson
% See also calcADC
%


ADC = double(ADC) ;

if nargin == 1
  ADCtop = 1700 ;
end


[ny nx nloca] = size(ADC) ;

% switch nloca
%  case 1
%   nr = 1 ; nc = 1;
%  case 2
%   nr = 2 ; nc = 1 ;
%  case 3
%   nr = 3 ; nc = 1 ;
%  case 4
%   nr = 2 ; nc = 2 ;
%  case 5
%   nr = 3 ; nc = 2 ;
%  case 6
%   nr = 3 ; nc = 2 ;
%  case 7
%   nr = 4 ; nc = 2 ;
%  case 8
%   nr = 4 ; nc = 2 ;
%  case 9
%   nr = 3 ; nc = 3 ;
%  case {10,11,12}
%   nr = 4 ; nc = 3 ;
%  otherwise
%   nr = round(sqrt(nloca)) ;
%   nc = ceil(nloca/nr) ;
% end

Dd = zeros([ny nx 1 nloca]) ;

for iloca = 1:nloca
  Dd(:,:,1,iloca) = mat2gray(ADC(:,:,iloca) ,[ 0 ADCtop]) ;
end

%figure('Name','ADC montage','Units','centimeters','Position',...
%       [3 3  fw fh])

%axes('Units','centimeters','Position',[0 0 fw fh])
hf = figure('Name','ADC montage') ;
%montage2(Dd,nr,nc)
prefs  = iptgetpref ;
iptsetpref('ImshowBorder','loose') ;
montage(Dd)

ticks = [ 0 0.25  0.5   0.75 1.0] ;
labs = ticks .* ADCtop  ;
labss = num2str(labs', '%4.0f') ;

%axes('Units','centimeters','Position',[16 2 1 22])
hc = colorbar('vert') ;
set(hc,'YTick',ticks) ;
set(hc,'YTickLabel',labss) ;
% set(hc,'Units','centimeters','Position',[16 2 1 22]);
% axes('Units','centimeters','Position',[16 2 1 22])
% axes(hc)
xlabel(' ADC x 10^{-3} mm^2 / s')

iptsetpref('ImshowBorder',prefs.ImshowBorder)
disp(['Opened figure: ',num2str(fignum(hf))])


 