classdef testverdict_fit < matlab.unittest.TestCase
    %TESTVERDICT_FIT Test for verdict_fit code
    % Allows pretty large errors for fVASC, fEES !
    %   
    %  Copyright 2022. David Atkinson. 
    

    methods(Test)
        function testvfitp(testCase)
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            import matlab.unittest.constraints.RelativeTolerance

            [gndt, scheme, Y, t, opt] = verdict_simulate ;
            
            tCell = namedargs2cell(t) ; optCell = namedargs2cell(opt) ;
            
            [fIC, fEES, fVASC, R, rmse, A, t, vfopt] = ...
                verdict_fit(scheme,Y, tCell{:}, optCell{:}) ;


            testCase.verifyThat( fIC, ...
                IsEqualTo(gndt.fIC, 'Within', AbsoluteTolerance(0.05)), ...
                'fICs too different') ;

            testCase.verifyThat( fEES, ...
                IsEqualTo(gndt.fEES, 'Within', AbsoluteTolerance(0.4)), ...
                'fEES too different') ;

            testCase.verifyThat( fVASC, ...
                IsEqualTo(gndt.fVASC, 'Within', AbsoluteTolerance(0.4)), ...
                'fVASC too different') ;

            testCase.verifyThat( R, ...
                IsEqualTo(gndt.R, 'Within', AbsoluteTolerance(2)), ...
                'R too different') ;

            testCase.verifyThat( A, ...
                IsEqualTo(gndt.A, 'Within', RelativeTolerance(0.01)), ...
                'A too different') ;

        end
    end
end

