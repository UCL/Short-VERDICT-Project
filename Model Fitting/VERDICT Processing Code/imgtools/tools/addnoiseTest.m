classdef addnoiseTest < matlab.unittest.TestCase
    % Tests of addnoise
    %
    % testCase = addnoiseTest ;
    % res = run( testCase ) ;
    %
    % Distribution checks need to be in air using Rayleigh or constant
    % signal region using Rician.
    %
    % David Atkinson  D.Atkinson@ucl.ac.uk
    %
    % See also SYSMATV ADDNOISE
    
    properties
        Testimg
    end
    
    methods( TestMethodSetup )
        % runs before each method
        function setvar(testCase)
            MRI = load('mri') ;
            testCase.Testimg = double(squeeze(MRI.D(:,:,1,12))) ;
        end
    end
    
    methods (Test)
        function testim2d(testCase)
            % Test noise on 2D image
            snr = 10 ;
            xout = addnoise(testCase.Testimg,'snr',snr) ;
            xair = abs(xout(1:30,1:30)) ;
            pd = fitdist(xair(:), 'Rician') ;
            testCase.verifyEqual(pd.sigma, 60/snr, 'RelTol',0.1)
        end
        
        function testim2db(testCase)
            % Similar to above - more inputs
            snr = 10 ;
            xout = addnoise(testCase.Testimg,'snr',snr,'indomain','image','signal',60) ;
            xair = abs(xout(1:30,1:30)) ;
            pd = fitdist(xair(:), 'Rayleigh') ;
            testCase.verifyEqual(pd.B, 60/snr, 'RelTol',0.1)
        end
        
        function testkprofile(testCase)
            % Add noise to k-space
            prof = testCase.Testimg(:,65) ;
            kin = i2k(prof) ;
            snr = 10 ;
            kout = addnoise(kin,'snr',snr,'signal',60,'indomain','kspace','scale','i2k') ;
            xout = k2i(kout) ;
            xair = abs(xout(1:15,1)) ;
            pd = fitdist(xair(:), 'Rayleigh') ;
            testCase.verifyEqual(pd.B, 60/snr, 'RelTol',0.3) % large tol due to limited samples
        end
        
        function testk0(testCase)
            % Add noise to k-space
            
            kin = zeros([128 1]) ;
            snr = 10 ;
            kout = addnoise(kin,'snr',snr,'signal',60,'indomain','kspace','scale','i2k') ;
            xout = k2i(kout) ;
            xair = abs(xout) ;
            pd = fitdist(xair(:), 'Rayleigh') ;
            testCase.verifyEqual(pd.B, 60/snr, 'RelTol',0.2) % 
        end
        
        function testk0b(testCase)
            % Add noise to larger k-space vector of zeros
            
            kin = zeros([1280 1]) ;
            snr = 10 ;
            kout = addnoise(kin,'snr',snr,'signal',60,'indomain','kspace','scale','i2k') ;
            xout = k2i(kout) ;
            xair = abs(xout) ;
            pd = fitdist(xair(:), 'Rayleigh') ;
            testCase.verifyEqual(pd.B, 60/snr, 'RelTol',0.1) % 
        end
        
        function testk0bal(testCase)
            % Test balanced scale using sysmatv to generate k-space and
            % lsqr to reconstruct.
            prof = 60*ones([1280 1]) ; 
           
            opt.scale = 'balanced';
            opt.verbose = false ;
            opt.vec = true;
            
            [yout, opt] = sysmatv(prof, 'notransp', opt) ;
            
            snr = 10 ; signal = 60 ;
            kout = addnoise(yout,'snr',snr,'signal',signal,'indomain','kspace','scale','balanced') ;
            
            fmv = @(x,flag)sysmatv(x, flag, opt) ;
            [xout] = lsqr(fmv, kout) ;
            
            pd = fitdist(abs(xout), 'Rician') ;
            
            testCase.verifyEqual(pd.sigma, signal/snr, 'RelTol',0.1) % 
            testCase.verifyEqual(pd.s, signal, 'RelTol', 0.1 ) ;
        end
        
    end
    
end
            