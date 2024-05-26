function [ I ]= Recon_withdB0_USM2(Sk,opt,arg1,arg2)

switch opt.recon_choice
    case 'cg'
        [~, I] = cg_Eh_Recon_USM2( Sk, arg1,arg2, opt );%
    case 'cp'
        I = Eh( Sk, arg1 );    
    case 'pwls_pcg'        
        I = pw_cg1_Eh_Recon_USM2( opt, 1, Sk, arg1,arg2 );
        
end

end
