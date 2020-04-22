function [batch_spec, batch_mz] = align_batch(xyzmats, mzs)
%%% inputs: XYZMATS should be a 1xN struct of desi images
%%%         MZS should be a NxM array of all mass-charge spectra
%%%         corresponding to the images in xyzmats
array_mz = mzs;
batch_mz = reshape(array_mz,1,[]); 
batch_mz = unique(round(batch_mz,1));
batch_mz = batch_mz(batch_mz~=0);
batch_mz = sort(batch_mz,'ascend');
batch_spec = {1:size(xyzmats,2)};
mz_size = size(batch_mz,2);
parfor mx = 1:size(xyzmats,2) %parallelized, can also just use regular for loop
    current_im = squeeze(cell2mat(xyzmats(mx)));
    [px, kx, zx] = size(current_im);
    mapped = zeros(px, kx, mz_size);
    for ix = 1:size(current_im,1)
        for jx = 1:size(current_im,2)
            pixel_spec = squeeze(current_im(ix,jx,:)).';
            count = remapping(pixel_spec, array_mz(mx,:), batch_mz);
            mapped(ix,jx,:) = count;
        end
    end
    batch_spec{mx} = mapped;
end
clear i

