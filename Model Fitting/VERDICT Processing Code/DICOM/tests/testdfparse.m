classdef testdfparse < matlab.unittest.TestCase
    %TESTDFPARSE Test for DFPARSE code
    %   
    %  Copyright 2021. David Atkinson. University College London
    
    methods(TestClassSetup)
        function prepareData(testCase)
            testCase.datafolder = '../testDICOMdata' ;
        end
    end

    properties
        datafolder
    end

    methods(Test)
        function testMultiFrame1(testCase)
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            import matlab.unittest.constraints.RelativeTolerance

            ffn = fullfile(testCase.datafolder,'T1W_FFE13slice.dcm') ;

            dinfo = dfparse(ffn) ;

            testCase.verifyThat(numel(dinfo), IsEqualTo(13))

            testCase.verifyThat(dinfo(13).ImageOrientationPatient, ...
                IsEqualTo([1 0 0 0 1 0]', 'Within', AbsoluteTolerance(0.09))) ;

            testCase.verifyThat(dinfo(4).FrameOfReferenceUID, ...
                IsEqualTo('1.3.46.670589.11.42055.5.0.22892.2020120417165006000')) ;

        end

        function testSingleFrame1(testCase)
            import matlab.unittest.constraints.IsEqualTo

            dfolder = fullfile(testCase.datafolder,'simulatedDicoms','*.dcm') ;

            dlist = dir(dfolder) ;
            for ifile = 1:length(dlist)
                fns{ifile} = fullfile(dlist(ifile).folder,dlist(ifile).name) ;
            end

            dinfo = dfparse(fns) ;

            testCase.verifyThat(numel(dinfo), IsEqualTo(8))

        end


    end

end

