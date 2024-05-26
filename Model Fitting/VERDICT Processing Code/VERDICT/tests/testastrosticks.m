classdef testastrosticks < matlab.unittest.TestCase
    %TESTASTROSTICKS Test for astrosticks code
    %   
    %  Copyright 2022. David Atkinson. 
    

    methods(Test)
        function testastrocomp(testCase)
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            import matlab.unittest.constraints.RelativeTolerance

            dVASC = 8 ;
            bval = 1500 ;
            sigexpected = 0.25583;

            sigcomputed = astrosticks(bval,dVASC) ;

            testCase.verifyThat( sigcomputed, ...
                IsEqualTo(sigexpected, 'Within', RelativeTolerance(0.01))) ;

        end

    end

end

