classdef teststejskal < matlab.unittest.TestCase
    %TESTSTEJSKAL Test for Stejskal code
    %   
    %  Copyright 2022. David Atkinson. University College London
    

    methods(Test)
        function testbfromg(testCase)
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            import matlab.unittest.constraints.RelativeTolerance

            delta = 23.9 ;
            Delta = 43.8 ;
            Gvalue = 32 ;
            bexpected = 1500 ;

            bcomputed = stejskal(delta, Delta, G=Gvalue) ;

            testCase.verifyThat(bcomputed, ...
                IsEqualTo(bexpected, 'Within', RelativeTolerance(0.1))) ;

        end

        function testgfromb(testCase)
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            import matlab.unittest.constraints.RelativeTolerance

            delta = 23.9 ;
            Delta = 43.8 ;
            Gexpected = 32 ;
            bvalue = 1500 ;

            Gcomputed = stejskal(delta, Delta, bval=bvalue) ;

            testCase.verifyThat(Gcomputed, ...
                IsEqualTo(Gexpected, 'Within', RelativeTolerance(0.1))) ;

        end


    end

end

