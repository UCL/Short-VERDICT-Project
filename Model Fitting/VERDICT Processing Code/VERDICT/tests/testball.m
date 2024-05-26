classdef testball < matlab.unittest.TestCase
    %TESTBALL Test for ball code
    %   
    %  Copyright 2022. David Atkinson. 
    

    methods(Test)
        function testballcomp(testCase)
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            import matlab.unittest.constraints.RelativeTolerance

            dEES = 2 ;
            bval = 1500 ;
            sigexpected = 0.049787 ;

            testCase.verifyThat(ball(0,dEES), ...
                IsEqualTo(1, 'Within', RelativeTolerance(0.01))) ;

            testCase.verifyThat(ball(bval,dEES), ...
                IsEqualTo(sigexpected, 'Within', RelativeTolerance(0.01))) ;

        end

    end

end

