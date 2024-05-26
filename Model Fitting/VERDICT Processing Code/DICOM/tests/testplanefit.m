classdef testplanefit < matlab.unittest.TestCase
    %TESTPLANEFIT Test plane_fit code
    %   
    %  Copyright 2021. David Atkinson. University College London

    % [NU, P, d] = plane_fit(coords3D)

    methods(Test)
        function testpointsax(testCase)
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            import matlab.unittest.constraints.RelativeTolerance

            % Axial plane LPH, all H constant, normal is in H
            coords3D = [ ...
                0   0 10 ;...
                10  1 10 ; ...
                10 10 10 ] ;

             [NU, P, d] = plane_fit(coords3D) ;

            testCase.verifyThat(abs(dot(NU,[0 0 1])), IsEqualTo(1, 'Within',RelativeTolerance(1e-5))) ;

            testCase.verifyThat(abs(d), IsEqualTo(10,'Within',RelativeTolerance(1e-6)))
        end

        function testpointssag(testCase)
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            import matlab.unittest.constraints.RelativeTolerance

            % Sag plane LPH, all L constant, normal is in L
            coords3D = ...
                [10 0 2 ;...
                 10 1 10 ; ...
                 10 4 10 ; ...
                 10 -6 3 ] ;

            [NU, P, d]  = plane_fit(coords3D) ;

            testCase.verifyThat(abs(dot(NU,[1 0 0])), IsEqualTo(1, 'Within',RelativeTolerance(1e-5))) ;

            testCase.verifyThat(abs(d), IsEqualTo(10,'Within',RelativeTolerance(1e-6)))

        end

        function testpointscor(testCase)
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            import matlab.unittest.constraints.RelativeTolerance

            % Axial plane LPH, all P constant, normal is in P
            coords3D = [ ...
                0 10 10 ;...
                2 10  4 ; ...
                9 10  1 ] ;

             [NU, P, d] = plane_fit(coords3D) ;

            testCase.verifyThat(abs(dot(NU,[0 1 0])), IsEqualTo(1, 'Within',RelativeTolerance(1e-5))) ;

            testCase.verifyThat(abs(d), IsEqualTo(10,'Within',RelativeTolerance(1e-6)))
        end

        function testpointsNotplanar(testCase)
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            import matlab.unittest.constraints.RelativeTolerance
            import matlab.unittest.constraints.IssuesWarnings

            coords3D = [ ...
                0 -10 10 ;...
                2 10  4 ; ...
                9 0  1 ; ...
                33 8 9 ] ;

            testCase.verifyThat(@() plane_fit(coords3D), ...
                IssuesWarnings({'MATLAB:plane_fit:NonCoplanar'}) )


        end

    end

end

