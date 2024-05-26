classdef bufferReconMoCo < handle & BaseBufferGadget
    
    properties
        
        image_num;
        series_num;
        
    end
    
    methods
        
        function g = config(g)
            
            %fprintf('The resonance frequency is %d\n', g.xml.experimentalConditions.H1resonanceFrequency_Hz);            
            g.image_num = 0;   % todo this needs to be static or global...
            g.series_num = 0;  % todo this needs to be static or global...
            
        end
        
        function g = process(g, recon_data)
            fprintf('Processing')
            
            %recon_data.data
            
            data = recon_data(1).data; %Assume only one encoding space
            head = data.headers; %Just get header from first trajectory
            
            img_data = data.data;
            
            %DA
            fn = tempname;
            save(fn,'recon_data')
            disp([' bufferReconMoCo recon_data saved to: ',fn])
            
            %DA Apply motion "correction"
            % could read in a file of motion parameters here, for now just
            % create a step motion hakf way through k-space. Readout
            % direction motion only.
            
            nfe = size(img_data,1) ;
            mparams = zeros([1 nfe]) ; % readout direction
            mparams(round(nfe/2):end) = 15 ; % 15mm step, half way through
            
            kx = [ 1: nfe ] ; % k-space coordinates
            
            kx = kx - double(recon_data.data.headers.center_sample(1)) ; % ?? correct usage??
            % DA added double above
            
            FOVx = recon_data.data.samplingdescription.recon_FOV(1) ;
            % Motion induced phases
            mphases = exp(-1i*2*pi*kx.*mparams/FOVx) ;
          
            % replicate motion phases in order to multiply with 
            % k-space data (called 'img_data' here)
            mphases = repmat(mphases',[1 size(img_data,2) size(img_data,3) size(img_data,4)]) ;
            
            img_data = img_data .* mphases ;
            
            
            % Inverse FFT
            %  use the 1/sqrt(N) convention for both fft and ifft
            %  so we need to scale the matlab/fftw functions
            %  which put all the scaling in the ifft
            
            % fft along x
            img_data = sqrt(size(img_data,1)) * fftshift(ifft(fftshift(img_data,1),[],1),1);
            
            % fft along y
            img_data = sqrt(size(img_data,2)) * fftshift(ifft(fftshift(img_data,2),[],2),2);
            
            % fft along z if 3D
            if size(img_data,3) > 1
                img_data =  sqrt(size(img_data,3)) * fftshift(ifft(fftshift(img_data,3),[],3),3);
            end

            % sqrt of sum of squares over channels
            img_data = sqrt(sum(abs(img_data).^2,4));
            
            % At the end of the acquisition, reconstruct the slice
            img_head = ismrmrd.ImageHeader;

            % set the matrix size
            % set one element at a time to not break the type (uint16) of matrix_size
            img_head.matrix_size(1) = size(img_data,1);
            img_head.matrix_size(2) = size(img_data,2);
            img_head.matrix_size(3) = size(img_data,3);

            img_head.position = head.position(:,1);
            img_head.read_dir = head.read_dir(:,1);
            img_head.phase_dir = head.phase_dir(:,1);
            img_head.slice_dir = head.slice_dir(:,1);
            img_head.patient_table_position = head.patient_table_position(:,1);
            img_head.acquisition_time_stamp = head.acquisition_time_stamp(1);
            img_head.image_index = g.image_num;
            img_head.image_series_index = g.series_num;
            img_head.channels = 1;
            
            %center = floor(size(img_data,3)/2)+1;
            %imagesc(abs(img_data(:,:,center,1,1,1,1))); axis image; axis square;
            %pause(2)
            %close()
            
            %disp(size(img_data));
            
            g.putImageQ(img_head, img_data);
            %fprintf('Put on Queue %d, type = %d\n',length(g.Q),g.Q{1}.type);
            
            
            
        end
        
    end
end
