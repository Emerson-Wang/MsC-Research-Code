files = dir('*.mat');
for ix = 1:size(files,1)
    load(files(ix).name);
    if ~exist('xyzmat_batch','var')
        xyzmat_batch = {xyzmat};
    else
        len = size(xyzmat_batch,2);
        xyzmat_batch{1,len+1} = xyzmat;
    end
    if ~exist('mcvec_batch','var')
        mcvec_batch = mcvec;
    else
        if size(mcvec,2) < size(mcvec_batch,2)
            mcvec = [mcvec,zeros(1, size(mcvec_batch,2)-size(mcvec,2))];
        end
        mcvec_batch = [mcvec_batch;mcvec];
    end
    disp(['File ', num2str(ix), ' loaded']);
end
disp('aligning')
[batch_spec, batch_mz] = align_batch(xyzmat_batch, mcvec_batch);
disp('saving')
save aligned_batch
