function epigeomraw

% Loop through the .data, .list files fro Sept 22, 2020 acquired with EPI
% and different offsets. Assumption is that the kspaces will all be the
% same.

pn =  '/Users/davidatkinson/OneDrive - University College London/data/20200922_phantoms/20200922_resphant/datalist' ;
if ~exist(pn,'dir')
    pn = uigetdir 
end

scan1 = fullfile(pn,'raw_036.data') ;
loca = 5 ;

MR1 = MRecon(scan1) ;
MR1 = recon2k(MR1) ;
     
img = sum(abs(k2i(MR1.Data(:,:,1,:,1,1,1,loca))),4) ;

figure
imgf = img-circshift(img,1) ;



% for ifile = 37:47
for ifile = [37:43 45:47]  % 44 is missing
    scan = fullfile(pn,['raw_0',num2str(ifile),'.data']) ;
    MR = MRecon(scan) ;
    MR = recon2k(MR) ;
    
    % eshow(k2i(MR.Data(:,:,1,:,1,1,1,loca)))
%     eshow(MR.Data(:,:,1,:,1,1,1,loca) - MR1.Data(:,:,1,:,1,1,1,loca), ...
%         'Name',num2str(ifile))
%     eshow(sum(abs(k2i(MR.Data(:,:,1,:,1,1,1,loca))) - abs(k2i(MR1.Data(:,:,1,:,1,1,1,loca))),4), ...
%         'Name',num2str(ifile))
    imgt = sum(abs(k2i(MR.Data(:,:,1,:,1,1,1,loca))) ,4) - img ; 
    imgf = cat(2,imgf, imgt) ;
end
eshow(imgf)


end %fn

function MR = recon2k(MR)

    switch MR.Parameter.DataFormat
        case{'ExportedRaw', 'Raw', 'Bruker' }
            
            %Reconstruct only standard (imaging) data
            MR.Parameter.Parameter2Read.typ = 1;
            MR.Parameter.Parameter2Read.Update;
            
            % Check if enough memory is available to reconstruct
            % the whole file at once. Otherwise reconstruct the
            % data in chunks
            if strcmpi(MR.Parameter.Recon.AutoChunkHandling, 'yes')
                [MemoryNeeded, MemoryAvailable, MaxDataSize] = MR.GetMemoryInformation;
                if MemoryNeeded > MemoryAvailable
                    if strcmpi( MR.Parameter.Recon.ImmediateAveraging, 'yes' ) || strcmpi( MR.Parameter.Recon.Average, 'yes' )
                        MR.Parameter.Chunk.Def = {'kx', 'ky', 'kz', 'chan', 'aver'};
                    else
                        MR.Parameter.Chunk.Def = {'kx', 'ky', 'kz', 'chan'};
                    end
                end
            end
            
            % Switch off for performance reasons (after recon it is
            % switched back to original state again)
            AutoUpdateStatus = MR.Parameter.Recon.AutoUpdateInfoPars;
            MR.Parameter.Recon.AutoUpdateInfoPars = 'no';
            
            % Define new counter
            counter = Counter( 'Performing Recon --> Chunk %d/%d\n');
            
            % Loop over all chunks
            for cur_loop = 1:MR.Parameter.Chunk.NrLoops
                
                % Update Counter
                if strcmpi( MR.Parameter.Recon.StatusMessage, 'yes')
                    counter.Update( {cur_loop ,MR.Parameter.Chunk.NrLoops} );
                end
                
                % Set the chunk-loop which automatically determines the
                % image parameter to read for the current chunk
                MR.Parameter.Chunk.CurLoop = cur_loop;
                
                % --------------------------------------------------------
                % Perform the Reconstruction for the Current Chunk (Start)
                % --------------------------------------------------------
                
                
                % imaging only ------------------------------
                MR.ReadData;
                
                MR.RandomPhaseCorrection;
                % MR.RemoveOversampling;
                MR.PDACorrection;
                MR.DcOffsetCorrection;
                MR.MeasPhaseCorrection;
                
                MR.SortData;
                edit 
%                 MR.GridData;
%                 MR.RingingFilter;
%                 MR.ZeroFill;
%                 MR.K2IM;
%                 MR.EPIPhaseCorrection;
%                 MR.K2IP;
%                 MR.GridderNormalization;
%                 MR.SENSEUnfold;
%                 MR.PartialFourier;
%                 MR.ConcomitantFieldCorrection;
%                 MR.DivideFlowSegments;
%                 MR.CombineCoils;
%                 MR.Average;
%                 MR.GeometryCorrection;
%                 MR.RemoveOversampling;
%                 MR.FlowPhaseCorrection;
%                 MR.ReconTKE;
%                 MR.ZeroFill;
%                 MR.RotateImage;
                
                
                % The current chunk is now reconstructed. If the data is
                % reconstructed in more than one chunk write the result to
                % a temporary file on the disk.
                if MR.Parameter.Chunk.NrLoops > 1
                    [exported_datafile, exported_listfile] = MR.WriteExportedRaw( [MR.Parameter.Filename.Data, '_temp.data'], MR.Parameter.Parameter2Read );
                end
                
                % --------------------------------------------------------
                % Perform the Reconstruction for the Current Chunk (End)
                % --------------------------------------------------------
            end
            
            % If data has been written to a temporary file read it
            % again
            if MR.Parameter.Chunk.NrLoops > 1
                r_temp = MRecon(exported_datafile);
                r_temp.ReadData;
                r_temp.Parameter.Recon.ImmediateAveraging = 'no';
                r_temp.SortData;
                MR.Data = r_temp.Data;
                fclose all;
                delete(exported_datafile);
                delete(exported_listfile);
                clear r_temp;
            end
            if strcmpi( MR.Parameter.Recon.StatusMessage, 'yes')
                fprintf('\n');
            end
            MR.Parameter.Recon.AutoUpdateInfoPars = AutoUpdateStatus;
            MR.Parameter.Reset;
        
        otherwise
            error( 'Error in Perform: Unknown data format' );
    end
end


        