
function scheme = BuildScheme(scans)
% Build a scheme structure
% scans(indx, delta, Delta, bval)


for scanIndx = 1:size(scans,1)

    delta = scans(scanIndx, 1);
    Delta = scans(scanIndx,2);
    bval = scans(scanIndx,3);
    TE = scans(scanIndx, 4);
    TR = scans(scanIndx, 5);
    Nav_ratio = scans(scanIndx, 6);

    scheme(scanIndx).delta = delta;
    scheme(scanIndx).DELTA = Delta ;
    scheme(scanIndx).bval = bval;
    scheme(scanIndx).G = stejskal(delta,Delta,bval=bval);
    scheme(scanIndx).TE = TE;
    scheme(scanIndx).TR = TR;
    scheme(scanIndx).Nav_ratio = Nav_ratio;
end
    
end