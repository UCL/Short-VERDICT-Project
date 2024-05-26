classdef testcoord2D3D < matlab.unittest.TestCase
    %TESTCOORD2D3D
    %   
    %  Copyright 2021-2022. David Atkinson. University College London

    % programmatically create roi
    % assign to next slice and check distances are slice separation
    %
    
    methods(TestClassSetup)
        function prepareData(testCase)
            info = dicominfo('../testDicomData/T1W_FFE13slice.dcm');
            dinfoF = dmfparse(info.Filename) ;
            [v,m,~] = d2mat(dinfoF,{'slice'},'op','fp') ;
            nslice = size(v,3) ;
            midslice = ceil(nslice/2) ;
            p1 = m.geom(midslice).IPP ;
            p2 = p1 + 10*m.geom(midslice).IOP(1:3) ;
            p3 = p2 + 10*m.geom(midslice).IOP(4:6) ;

            cont10px{1} = [ p1(:)' ; p2(:)' ; p3(:)' ] ;

            % In image (1,1) (1,11), (11,11) in (x,y)
            p5 = p1 + 10*m.geom(midslice).PixelSpacing_HW(2) * m.geom(midslice).IOP(1:3) ;
            p6 = p5 + 10*m.geom(midslice).PixelSpacing_HW(1) * m.geom(midslice).IOP(4:6) ;

            cont10mm{1} = [ p1(:)' ; p5(:)' ; p6(:)' ] ;
            
            testCase.refInfo = info ;
            testCase.contours10mm = cont10mm ;
            testCase.contours10px = cont10px ;

            testCase.midsl = midslice;
            testCase.geom  = m.geom ;
            testCase.dinfo = dinfoF ;

        end
    end

    properties
        refInfo
        contours10mm
        contours10px
        midsl
        geom
        dinfo
    end

    methods(Test)
        function test3Dto2D(testCase)
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            import matlab.unittest.constraints.RelativeTolerance

            %  [coord2D, distances] = coord2D3D(coord3D, geom) 

            coord3D = testCase.contours10mm{1} ;
            
            [coord2D, distances] = coord2D3D(coord3D, testCase.geom(testCase.midsl)) ;

            % check output sizes
            testCase.verifyThat(size(coord2D,1), IsEqualTo(size(coord3D,1))) 
            testCase.verifyThat(numel(distances), IsEqualTo(size(coord3D,1)))

            testCase.verifyThat(size(coord2D,2), IsEqualTo(2)) 

            % check distances are zero for these contours (in plane of
            % slice)
            testCase.verifyThat(sum(distances),IsEqualTo(0 , 'Within', AbsoluteTolerance(1e-6)) )


            % repeat for one slice away
            [coord2D1sl, distances1sl] = coord2D3D(coord3D, testCase.geom(1+testCase.midsl)) ;

            % 2D coords should be the same as previous slice
            testCase.verifyThat(coord2D1sl, IsEqualTo(coord2D, 'Within',RelativeTolerance(1e-5)) )

            % all distances should be slice centre separation
            expecteddist = repmat(testCase.dinfo(testCase.midsl).slc2c, size(distances1sl)) ;

            testCase.verifyThat(distances1sl, IsEqualTo(expecteddist,'Within',AbsoluteTolerance(1e-5)))

        end

        function test2Dto3D(testCase)
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            import matlab.unittest.constraints.RelativeTolerance

            %  [coord3D ] = coord2D3D(coord2D, geom) 

            coord2D = [1 1 ; 11 1 ; 11 11] ;
            geo_this = testCase.geom(testCase.midsl) ;
 
            [ coord3D ] = coord2D3D(coord2D, geo_this) ;

            % check output sizes
            testCase.verifyThat(size(coord3D,1), IsEqualTo(size(coord2D,1))) 

            testCase.verifyThat(size(coord3D,2), IsEqualTo(3)) 

            testCase.verifyThat(coord3D, IsEqualTo(testCase.contours10mm{1}, 'Within',AbsoluteTolerance(1e-5)))

        end

        function testRoundTrip1(testCase)
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            import matlab.unittest.constraints.RelativeTolerance

            coord2D = [1 1 ; 11 1 ; 11 11] ;
            geo_this = testCase.geom(testCase.midsl) ;
 
            [ coord3D ] = coord2D3D(coord2D, geo_this) ;

            [coord2Drt, distances] = coord2D3D(coord3D, geo_this) ;

            testCase.verifyThat(coord2Drt, IsEqualTo(coord2D,'Within', RelativeTolerance(1e-5)))

            testCase.verifyThat(sum(abs(distances)), IsEqualTo(0,'Within', AbsoluteTolerance(1e-4))) 

        end

        function testRoundTrip2(testCase)
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            import matlab.unittest.constraints.RelativeTolerance

            coord2D = [1 1 ; 11 1 ; 11 11] ;
            geo_this = testCase.geom(testCase.midsl) ;
 
            [ coord3D ] = coord2D3D(coord2D, geo_this, 'type2D', '2Dintrinsic') ;

            [coord2Dmm] = coord2D3D(coord3D, geo_this, 'type2D', '2Dmm') ;

            [ coord3D ] = coord2D3D(coord2Dmm, geo_this, 'type2D', '2Dmm') ; 

            [coord2Dintrinsic, ~] = coord2D3D(coord3D, geo_this, 'type2D', '2Dintrinsic') ;

            testCase.verifyThat(coord2Dintrinsic, IsEqualTo(coord2D,'Within', RelativeTolerance(1e-5)))

        end

        function testAxpoints(testCase)
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            import matlab.unittest.constraints.RelativeTolerance

            coord2Dintrinsic = [1 1 ; 1 3 ; 3 3; 3 1] ;
            coord2Dmm        = [0 0 ; 0 4 ; 4 4; 4 0] ;
            coord3D          = [0 0 10 ; 0 4 10 ; 4 4 10 ; 4 0 10 ] ;

            geomt.IOP = [1 0  0 0 1 0] ; % IOP axial
            geomt.IPP = [0 0 10] ;
            geomt.PixelSpacing_HW = [2 2] ;
            geomt.XData(1) = 0 ;
            geomt.YData(1) = 0 ; 

 
            [ coord3Dpred ] = coord2D3D(coord2Dintrinsic, geomt, 'type2D', '2Dintrinsic') ;
            testCase.verifyThat(coord3Dpred, IsEqualTo(coord3D,'Within', RelativeTolerance(1e-5))) ;

            [ coord3Dpredb ] = coord2D3D(coord2Dmm, geomt, 'type2D', '2Dmm') ;
            testCase.verifyThat(coord3Dpredb, IsEqualTo(coord3D,'Within', RelativeTolerance(1e-5))) ;

            [coord2Dmmpred] = coord2D3D(coord3D, geomt, 'type2D', '2Dmm') ;
            testCase.verifyThat(coord2Dmmpred, IsEqualTo(coord2Dmm,'Within', RelativeTolerance(1e-5))) ;

            [coord2Dintpred] = coord2D3D(coord3D, geomt, 'type2D', '2Dintrinsic') ; 
            testCase.verifyThat(coord2Dintpred, IsEqualTo(coord2Dintrinsic,'Within', RelativeTolerance(1e-5))) ;

        end
        

    end

end

