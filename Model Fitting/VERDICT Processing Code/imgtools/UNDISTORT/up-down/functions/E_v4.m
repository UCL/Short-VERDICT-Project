function [u] = E_v4(v,arg)
% Encoding function with dB0 phase term and coil sensitivities
%
%inputs
S        = arg.S;
t_s      = arg.t_s;
dB0_rads = arg.dB0_rads;
kx       = arg.k.x;
ky       = arg.k.y;

[sy,sx,nch,~] = size(v);

if isfield(arg,'samp')
    samp = arg.samp;
else
    samp = ones(sy,sx);
end

%initialise
u = zeros(sy,sx,nch);

%k-space co-ord system
x_coord = (1:sx) - (sx / 2) - 1;
y_coord = (1:sy) - (sy / 2) - 1;

        tic
        num_time_points = numel(t_s);
        x_coord_mat = repmat(x_coord, [sy 1]);
        y_coord_mat = repmat(y_coord(:),[1 sx]);
        y_coord_mat = y_coord_mat(:);
        x_coord_mat = x_coord_mat(:);
        dB0_rads = dB0_rads(:);
        
        u = zeros(sy*sx,nch);
        size(v)
        size(S)
        v = v .* S;
        for chn = 1:nch
            v_c = v(:,:,chn);
            v_c = v_c(:);
            parfor i = 1:num_time_points %par
                % Phase due to encoding gradients and B0 inhomogeneities
                phase_pix = ( kx(i) .* x_coord_mat ) + ( ky(i) .* y_coord_mat ) + ( t_s(i) .* dB0_rads ); %               
                
                sol =  v_c.' * exp( -1i .* phase_pix ); 
                u(i,chn) = sol;
            end
        end
        u = reshape(u,[sy sx nch]);
        toc
    

% Fix intensities (due to summation)
u = u ./ sqrt(sx .* sy);

%multiply by sampling matrix
u = repmat(samp,[1 1 nch]) .* u;
end