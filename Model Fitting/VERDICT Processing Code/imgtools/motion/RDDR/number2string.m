function numstr=number2string(num)
    num=num2str(num);
    numzeros='0000';  
    numstr=[numzeros(length(num):end) num];