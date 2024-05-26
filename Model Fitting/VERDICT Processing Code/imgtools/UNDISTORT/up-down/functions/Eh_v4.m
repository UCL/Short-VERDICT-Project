function [v] = Eh_v4(u,arg)
% Inverse encoding function with dB0 phase term and coil sensitivities
% 

data_type = @double;

%inputs
S        = data_type(arg.S);
t_s      = data_type(arg.t_s(:));
dB0_rads = data_type(arg.dB0_rads);
kx       = data_type(arg.k.x);
ky       = data_type(arg.k.y);
u        = data_type(u);

[sy,sx,nch,~] = size(u);

%k-space co-ord system
x_coord = data_type( (1:sx) - (sx / 2) - 1 );
y_coord = data_type( (1:sy) - (sy / 2) - 1 );
Sc= conj(S);
      
         
        num_pixels = sy*sx;
        x_coord_mat = repmat(x_coord, [sy 1]);
        y_coord_mat = repmat(y_coord(:),[1 sx]); 
        kx = kx(:);
        ky = ky(:);        
        t_s = t_s(:);
        
        v = zeros(sy*sx,nch);
        for chn = 1:nch
            u_c = u(:,:,chn);
            u_c = u_c(:);
            parfor pix = 1:num_pixels  %par
                % Phase due to encoding gradients and B0 inhomogeneities
                pix_phase = ( kx .* x_coord_mat(pix) ) + ( ky .* y_coord_mat(pix) ) + ( t_s .* dB0_rads(pix) ); % this is correct, checked against mex function
                
                sol =  u_c.' * exp( 1i.* pix_phase ) ; 
                
                v(pix,chn) = sol; %crazily doing this is faster!
            end
        end
        v = sum(reshape(v,[sy sx nch]) .* Sc,3);
       % v=v./(sqrt(sum(abs(Sc).^2,3))+0.005);
    
v = v ./ sqrt(sx .* sy);
v(isnan(v))=0;
end