
classdef sysmatvTest < matlab.unittest.TestCase
    % Tests of sysmatv
    % See also demo_sysmatv
    
    properties
        TestxinMRI
        Testxinprostate
    end
    
    methods( TestMethodSetup )
        % runs before each method
        function setvar(testCase)
            D = load('mri') ;
            xin = double(D.D(65,:,1,10)) ; % use line through mri slice
            testCase.TestxinMRI = xin(:) ;
            
            img = prostate(128) ;
            % profiles through prostate and rectum
            xin = img(:,60) ;
            testCase.Testxinprostate = xin ;
        end
    end
    
    methods (Test)
        function testpF(testCase)
            % Test partial Fourier is zeroing correctly
            
            opt.verbose = true ;
            opt.scale = 'i2k' ;
            opt.partialFReq = 0.7 ;
            
            xin = testCase.TestxinMRI ;
            
            [yout, optout] = sysmatv(xin, 'notransp', opt) ;
            
            % Check zeroed k-space
            nzexpect = (1-optout.partialFAct) * 128 ; 
            testCase.verifyEqual( norm(abs(yout(1:nzexpect))), 0, 'AbsTol', 1e-5)
            % Check i2k is same as remaining yout k-space
            kin = i2k(xin) ;
            testCase.verifyEqual( norm(abs(yout(nzexpect+1:end))-abs(kin(nzexpect+1:end))), 0, 'AbsTol', 1e-5) 
        end % testpF
        
        function testckstartend(testCase)
            % Check k-space coordinates set up OK when nR > 1
            xin = testCase.TestxinMRI ;
            
            opt.Rreq = [1 2] ;
            
            [yout, optout] = sysmatv(xin, 'notransp', opt) ;
            testCase.verifyEqual( optout.ckstart(1),1) ;
            testCase.verifyEqual( optout.ckstart(2),129) ;
            testCase.verifyEqual( optout.ckend(1),128) ;
            testCase.verifyEqual( optout.ckend(2),128+64) ;
            testCase.verifyEqual( length(yout), 128+64) ;
            testCase.verifyEqual( optout.nkperR, [ 128 64] ) ;
        end
        
        function testcomb(testCase)
            xin = testCase.TestxinMRI ;
            
            opt.Rreq = [1 2] ;
            opt.partialFReq = 0.75 ; % should haveno effect on k-space coordinates 
            opt.csens = ones([128 3]) ; % 3 coils
            opt.vec = false ;
            
            [yout, optout] = sysmatv(xin, 'notransp', opt) ;
            [xout, optout] = sysmatv(yout, 'transp', optout) ;
            
            szyout = size(yout) ;
            testCase.verifyEqual( szyout(1), 128+64 )
            testCase.verifyEqual( szyout(end), 3 )
            testCase.verifyEqual( optout.Ract, [1 2] ) % 128 divides into [1 2]
            
            testCase.verifyEqual( size(xout), size(xin) )
        end
        
    end
    
end