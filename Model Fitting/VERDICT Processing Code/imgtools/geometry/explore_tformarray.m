function explore_tformarray
% explore tformarray function

%   B = TFORMARRAY(A,T,R,TDIMS_A,TDIMS_B,TSIZE_B,TMAP_B,F) applies
%   a geometric transformation to array A to produce array B.

% We are going to apply a 3D transformation so TDIMS_A should be length 3
% with values 1,2 and 3, but not necessarily in order. 
% TDIMS_B should also be similar, e.g. [1 2 3]
% TMAP_B  can be of size [size(A,1) size(A,2) size(A,3) L] where L is 
% length(TDIMS_A)

% For DICOM resampling, we need to compute the coordinates in A (in image
% space) of the required points in B. We know where the 3D points are in A,
% and where we want the points for B - interpn would be easier were it not
% for the requirement here to use makeresampler.
%
% 


%A = zeros(20,10,5) ;
% A = gallery('integerdata',20,[20, 10, 5],1) ;

load mri
A = squeeze(double(D)) ;

% ndgrid expects input and output in order x,y,z
% Using [D1, D2, D3] = ndgrid( 1:size(A,1), 1:size(A,2), 1:size(A,3)) 
% can be confusing as size(A,1) is for the row dimension.
[D1, D2, D3] = ndgrid( 1:size(A,1), 1:size(A,2), 1:size(A,3)) ;

T = [] ; % using TMAP_B instead 

R = makeresampler('cubic','fill') ;

TDIMS_A = [1 2 3] ;

TDIMS_B = [1 2 3] ;

TSIZE_B = [] ; % empty as specified by first dimensions of TMAP_B

F = 0 ; % fill values

szB = [size(A,1) size(A,2) size(A,3)] ; % output size of B, possibly permuted by TDIMS_B

TMAP_B = zeros(szB(1), szB(2), szB(3), 3) ;

for id1 = 1:szB(1)
    for id2 = 1:szB(2)
        for id3 = 1:szB(3)
            % coordinate of point in A to be re-sampled, assuming top left centre is
            % 1,1,1.
            % This puts at point [id1, id2m id3] the value interpolated
            % from A at the value in coordA. Note these are in MATLAB image
            % row col slice coords. 
            % So, adding 10 to the first dimension means that a point 10
            % pixels of A lower down is interpolated and placed at the
            % current coord in B [id1, id2, id3]. In this example, the
            % result in B appears shifted upwards by 10 pixels.
            
            coordinA = [ D1(id1,id2,id3)+10 D2(id1, id2, id3) D3(id1,id2,id3)] ;
            TMAP_B(id1,id2,id3,:) = coordinA ;
        end
    end
end


B = tformarray(A,T,R,TDIMS_A,TDIMS_B,TSIZE_B,TMAP_B,F) ;

eshow(A)
eshow(B)

