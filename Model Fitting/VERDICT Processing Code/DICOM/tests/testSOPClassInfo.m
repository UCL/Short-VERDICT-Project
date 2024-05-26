classdef testSOPClassInfo < matlab.unittest.TestCase
    %TESTSOPClassInfo
    %   
    %  Copyright 2021. David Atkinson. University College London
    
    
    methods(Test)
        function testOutput(testCase)
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            import matlab.unittest.constraints.RelativeTolerance

            SOP = SOPClassInfo ;

            % Should have fields UID, code and desc 
            expectedfields = {'UID','code','desc'}' ;
            testCase.verifyThat(fieldnames(SOP), IsEqualTo(expectedfields)) 

        end

    end

end

