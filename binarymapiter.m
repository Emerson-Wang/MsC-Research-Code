function [idx] = binarymapiter(value, mapvec)
    idx = 0;
    collect = 0;
    index = 1:size(mapvec,2);
    while(size(mapvec,2)>3)
        ix = floor(size(mapvec,2)/2);
        if(value==mapvec(ix))
            idx = index(ix);
            return;
        elseif(value>mapvec(ix))
            mapvec = mapvec(ix:end);
            index = index(ix:end);
        elseif(value<mapvec(ix))
            mapvec = mapvec(1:ix);
            index = index(1:ix);
        end
    end
    min = Inf;
    for jx = 1:size(mapvec,2)
       diff = abs(value-mapvec(jx));
       if diff < min
           min = diff;
           idx = index(jx);
       end
   end