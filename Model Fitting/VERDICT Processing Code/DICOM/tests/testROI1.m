classdef testROI1 < matlab.unittest.TestCase
    %TESTROI1 Test for RTStruct code
    %   
    %  Copyright 2021. David Atkinson. University College London
    
    methods(TestClassSetup)
        function prepareData(testCase)
            info = dicominfo('../testDicomData/T1W_FFE13slice.dcm');
            dinfo = dmfparse(info.Filename) ;
            [v,m,l] = d2mat(dinfo,{'slice'},'op','fp') ;
            nslice = size(v,3) ;
            midslice = ceil(nslice/2) ;
            p1 = m.geom(midslice).IPP ;
            p2 = p1 + 10*m.geom(midslice).IOP(1:3) ;
            p3 = p2 + 10*m.geom(midslice).IOP(4:6) ;

            cont10mm{1} = [ p1(:)' ; p2(:)' ; p3(:)' ] ;

            p5 = p1 + 10*m.geom(midslice).PixelSpacing_HW(2) * m.geom(midslice).IOP(1:3) ;
            p6 = p5 + 10*m.geom(midslice).PixelSpacing_HW(1) * m.geom(midslice).IOP(4:6) ;

            cont10px{1} = [ p1(:)' ; p5(:)' ; p6(:)' ] ;
            
            testCase.refInfo = info ;
            testCase.contours10mm = cont10mm ;
            testCase.contours10px = cont10px ;

        end
    end

    properties
        refInfo
        contours10mm
        contours10px
    end

    methods(Test)
        function testTriang(testCase)
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            import matlab.unittest.constraints.RelativeTolerance

            
            RTcont = dicomRTStruct(testCase.refInfo) ;
            
            num = 1; Name = 'tri10mm'; geom_type = 'CLOSED_PLANAR'; col = [255 0 0];
            RTcont = addContour(RTcont,num, Name, testCase.contours10mm, geom_type, col) ;

            num = 2; Name = 'tri10pix'; geom_type = 'CLOSED_PLANAR'; col = [0 0 255];
            RTcont = addContour(RTcont,num, Name, testCase.contours10px, geom_type, col) ;

            RTInfo = convertToInfo(RTcont) ;

            % 3D Slicer only takes a folder as input so make one
            mkdir(tempdir,'DICOMtestfiles')
            dfn = 'Triang.dcm' ;
            ffn = fullfile(tempdir,'DICOMtestfiles',dfn) ;
            status = dicomwrite([],ffn,RTInfo,'CreateMode','copy') 


            disp(['Visually check this file: '])
            disp(['   ', ffn])
            disp('Then issue the command:')
            disp(['  delete(''',ffn,''')'])

        end

    end

end

