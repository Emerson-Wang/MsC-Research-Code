function [tissue] = tissue_region(xyzmat, tissueLabel)
[m,n,k] = size(xyzmat);
tissue = [];
for ix = 1:m
    for jx = 1:n
        if(tissueLabel(ix,jx) == 1)
            tissue = [tissue; xyzmat(ix,jx,:)];
        end
    end
end