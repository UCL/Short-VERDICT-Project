function dinfo_out = dinfostrip(dinfo_in)
% DINFOSTRIP Strip out DICOM info data, keeping only limited tags
%
% dinfo_out = dinfostrip(dinfo_in)
%
% Example:
%
% metadata = dicominfo('CT-MONO2-16-ankle.dcm');
% X = dicomread(metadata) ;
% anonmetadata = dinfostrip(metadata) ;
% dicomwrite(X, 'ct_file.dcm', anonmetadata);
%
% $Id: dinfostrip.m 314 2011-11-14 10:27:58Z ucacdat $
% David Atkinson, UCL.  D.Atkinson@ucl.ac.uk
% 
% See also DICOMANON DICOMAN

% Note Philips Private 2005 140f  contains more MR parameters but is not
% copied here.
% Private_2005_100e (scaling to floating point) is saved.

tokeep = { ...
    'AcquisitionContrast'
    'AcquisitionDuration'
     'AcquisitionMatrix'
     'AcquisitionTime'
    'BitDepth'
    'BitsAllocated'
    'BitsStored'
    'BodyPartExamined'
    'ColorType'
    'Columns'
    'ContentTime'
    'dBdt'
    'DiffusionBValue'
    'DiffusionGradientOrientation'
    'EchoNumbers'
    'EchoTime'
    'EchoTrainLength'
    'FileSize'
    'Filename'
    'FlipAngle'
    'Format'
    'FormatVersion'
    'HeartRate'
    'Height'
    'HighBit'
    'ImageOrientationPatient'
    'ImagePositionPatient'
    'ImageType'
    'ImagedNucleus'
    'ImagingFrequency'
    'InPlanePhaseEncodingDirection'
    'InstanceNumber'
    'MagneticFieldStrength'
    'Modality'
    'NumberOfAverages'
    'NumberOfPhaseEncodingSteps'
    'NumberOfTemporalPositions'
    'PatientPosition'
    'PercentPhaseFieldOfView'
    'PercentSampling'
    'PhotometricInterpretation'
    'PixelBandwidth'
    'PixelRepresentation'
    'PixelSpacing'
    'Private_2005_100e'
    'Private_2005_1030'
    'ProtocolName'
    'ReceiveCoilName'
    'ReconstructionDiameter'
    'RepetitionTime'
    'RescaleIntercept'
    'RescaleSlope'
    'RescaleType'
    'Rows'
    'SAR'
    'SOPClassUID'
    'SamplesPerPixel'
    'ScanningSequence'
    'SeriesDescription'
    'SeriesNumber'
    'SeriesTime'
    'SliceLocation'
    'SliceThickness'
    'SpacingBetweenSlices'
    'StudyDescription'
    'StudyTime'
    'TemporalPositionIdentifier'
    'TransmitCoilName'
    'TriggerTime'
    'Width'
    'WindowCenter'
    'WindowWidth' } ;



for ifield = 1:length(tokeep)
    if isfield(dinfo_in, tokeep(ifield))
      dinfo_out.(tokeep{ifield}) = dinfo_in.(tokeep{ifield}) ;
    end
end
