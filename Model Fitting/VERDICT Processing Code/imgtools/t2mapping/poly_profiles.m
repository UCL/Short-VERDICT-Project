function poly_profiles(hfig, slc, ect, v)

figure(hfig)
hrois = findobj(hfig,'Type','images.roi.polygon') ;

np = length(hrois) ;

figure('Name',['ROI from fig: ',num2str(hfig.Number)])
for ip = 1:np
    bw = createMask(hrois(ip)) ;
    se = zeros([1 length(ect)]) ;
    for ie = 1:length(ect)
        img = v(:,:,slc,ie) ;
        vals = img(bw) ;
        se(ie) = median(vals(:)) ;
    end
    
    lab = get(hrois(ip),'Label') ;
    plot(ect,se,'DisplayName',lab), hold on, grid on
end
legend
