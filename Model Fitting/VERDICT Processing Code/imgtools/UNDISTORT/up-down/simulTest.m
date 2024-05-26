function tests = simulTest
% SIMULTEST for testing simulations in UNDISTORT
% Tests for joint_simul. Tests are selected from a listbox of those within
% this file.
%
% tests = runtests('simulTest.m') ;
%
% D.Atkinson@ucl.ac.uk
%
% See also SIMULSET JOINT_SIMUL

% Find functions in this .m file and put in listbox for user selection
fh = localfunctions ;
for istr = 1:length(fh)
    lbstr{istr} = func2str(fh{istr}) ;
end

hf = figure('Name','Select tests') ;
hlb = uicontrol(hf,'Style','listbox','Max',2,'Min',0, ...
    'Position',[30 70 300 300], 'FontSize', 12, ...
    'String',lbstr, 'Callback',@listbox_callback) ;
hpb = uicontrol(hf,'Style','pushbutton', 'String','Run selected',...
    'FontWeight','bold','FontSize', 14,'Callback','delete(gcf)', ...
    'Position',[30 15 300 40]) ;

testsl = 1 ; % Will be overwritten on figure closure

uiwait(hf) % Wait for user to press button 

tests = functiontests({fh{testsl}});

    function listbox_callback(h, event)
        testsl = get(h,'Value') ;
    end

end



function testSENSEpF(testCase)
% Non-integer SENSE, partial Fourier. Exact B0
simulopt = simulset('name', 'SENSEpF', 'npe_u', 35, 'npe_f', 63 ,'nfe', 64, 'partialF', 0.7, ...
    'csens_obj_only', true, 'ncoil', 4, 'B0_amp',100, 'time_seg', true, ...
    'B0_joint', false, 'x0_joint', true, 'addnoise', true, 'B0_scale', 1., 'B0_offset', 0, ...
    'x0', 'empty', 'check_linear', true , 'B0_project_real', true,  ...
    'cph_mult',10, 'es',0*[2.3/2 -2.3/2]*1e-3, 'nouter',1, 'ntraj', 2,'maxit',10 );

rng('default')
ceiling = 0.05 ; %0.05 if noise added
actual = joint_simul(simulopt);
import matlab.unittest.constraints.IsLessThan;
testCase.verifyThat(actual, IsLessThan(ceiling));
end

function testfminsearch(testCase)
% Detects global linear error in B0 (scale + constant offset)
simulopt = simulset('name', 'fminsearch', 'npe_u', 63, 'npe_f', 63 ,'nfe', 64, 'partialF', 1, ...
    'csens_obj_only', false, 'ncoil', 1, 'B0_amp', 50, 'time_seg', true, ...
    'B0_joint', true, 'x0_joint', true, 'addnoise', true, 'B0_scale', 1.1, 'B0_offset', 20, ...
    'x0', 'empty', 'check_linear', false , 'B0_project_real', true,  ...
    'es',0*[2.3/2 -2.3/2]*1e-3, 'nouter',10, 'ntraj', 2,'maxit',10, 'B0_opt','fminsearch');
rng('default')
ceiling = 0.1 ;
actual = joint_simul(simulopt);
import matlab.unittest.constraints.IsLessThan;
testCase.verifyThat(actual, IsLessThan(ceiling));
end



function testSENSEpFtscheck(testCase)
% Time segmentation and check (afun), SENSE and partial Fourier
simulopt = simulset('name', 'SENSEpFts', 'npe_u', 19, 'npe_f', 31 ,'nfe', 32, 'partialF', 0.8, ...
    'csens_obj_only', true, 'ncoil', 4, 'B0_amp',00, 'time_seg', false, 'x_check_tseg', true, ...
    'B0_joint', false, 'x0_joint', true, 'addnoise', false, 'B0_scale', 1., 'B0_offset', 0, ...
    'x0', 'empty', 'check_linear', false , 'B0_project_real', true,  ...
    'cph_mult',0, 'es',0*[2.3/2 -2.3/2]*1e-3, 'nouter',1, 'ntraj', 2,'maxit',10 );

rng('default')
ceiling = 0.01 ; %0.05 if noise added
actual = joint_simul(simulopt);
import matlab.unittest.constraints.IsLessThan;
testCase.verifyThat(actual, IsLessThan(ceiling));
end


function testU2px(testCase)
% Residual falls slightly but true error grows with iterations. Test
% 'fails'
simulopt = simulset('name', 'U4px', 'npe_u', 111, 'npe_f', 111 ,'nfe', 112, 'partialF', 0.7, ...
    'csens_obj_only', false, 'ncoil', 1, 'B0_amp',10, 'time_seg', true, ...
    'B0_joint', true, 'x0_joint', true, 'addnoise', false, 'B0_scale', 1., 'B0_offset', 25 , ...
    'x0', 'empty', 'check_linear', true , 'B0_project_real', true,  ...
    'es',0*[2.3/2 -2.3/2]*1e-3, 'dt_s',0.7e-3, 'nouter',25, 'ntraj', 2,'maxit',10 );

rng('default')
ceiling = 0.1 ;
actual = joint_simul(simulopt);
import matlab.unittest.constraints.IsLessThan;
testCase.verifyThat(actual, IsLessThan(ceiling));
end


function testJointwB0wnoise(testCase)
% Residual falls slightly but true error grows with iterations
simulopt = simulset('name', 'JointwB0noise', 'npe_u', 63, 'npe_f', 63 ,'nfe', 64, 'partialF', 1, ...
    'csens_obj_only', false, 'ncoil', 4, 'B0_amp',300, 'time_seg', true, ...
    'B0_joint', true, 'x0_joint', true, 'addnoise', true, 'B0_scale', 1., 'B0_offset', 0, ...
    'x0', 'empty', 'check_linear', false , 'B0_project_real', true,  ...
    'es',0*[2.3/2 -2.3/2]*1e-3, 'nouter',5, 'ntraj', 2,'maxit',10 );

rng('default')
ceiling = 0.1 ;
actual = joint_simul(simulopt);
import matlab.unittest.constraints.IsLessThan;
testCase.verifyThat(actual, IsLessThan(ceiling));
end






