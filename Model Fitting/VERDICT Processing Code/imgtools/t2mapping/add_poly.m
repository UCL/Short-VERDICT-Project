function add_poly(ha)
another = true ;
while another == true
    hroi = drawpolygon(ha,'FaceAlpha',0) ;
    
    lab = input('Label: ','s') ;
    set(hroi,'Label',lab)
    
    cont = input('Continue? 0=No, 1=Yes: ') ;
    if cont == 1
        another = true ;
    else
        another = false ;
    end
end