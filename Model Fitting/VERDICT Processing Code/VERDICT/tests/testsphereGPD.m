classdef testsphereGPD < matlab.unittest.TestCase
    %TESTSPHEREGPD Test for astrosticks code
    %   
    %  Copyright 2022. David Atkinson. 
    

    methods(Test)
        function testsphereGPDcomp(testCase)
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            import matlab.unittest.constraints.RelativeTolerance

            delta = 10 ;
            DELTA = 50 ;
            G = 40 ;
            r = 10 ;
            d = 2 ;
            sigexpected = 0.84324; % just from running function, not an indpendent test

            sigcomputed = sphereGPD(delta, DELTA, G, r, d) ;

            testCase.verifyThat( sigcomputed, ...
                IsEqualTo(sigexpected, 'Within', RelativeTolerance(0.01))) ;

        end

    end

end

