% PREPROCESS is a script that performs tissue selection 
% using tissue region masks, pixel selection using NMF, and ion selection.
tissue_xyz = {1:size(align_xyz,2)};
k = 10;
parfor ix = 1:size(align_xyz,2) % reduce samples to tissue regions, then perform pixel selection with NMF
    tissue_xyz{ix} = tissue_region(cell2mat(align_xyz(ix)), cell2mat(tissueRegions(ix)));
    current_xyz = squeeze(tissue_xyz{ix});
    [W, ~] = nnmf(current_xyz.', k);
    score = [];
    for jx = 1:size(current_xyz,1)
        sc = current_xyz(jx,:)*W;
        score = [score;sc];
    end
    new_xyz = score;
    norm_xyz = vecnorm(new_xyz,2,2);
    [norm_xyz, ind] = sort(norm_xyz,'descend');
    current_xyz = current_xyz(ind,:);
    sample_n = floor(size(new_xyz,1)/2.5);
    tissue_xyz{ix} = current_xyz(1:sample_n,:);
end
[final_xyz] = ion_pick(align_mz,tissue_xyz,ions); % perform ion selection

% Exploratory options
%[final_xyz, flat_batch] = peak_pick_and_PCA(align_mz,tissue_xyz,40);
%[final_xyz,flat_batch] = peak_pick_and_nmf(align_mz,tissue_xyz, 10, 10);
%[pca_set] = comp_analysis(align_mz,tissue_xyz, 3);
%[final_xyz] = image_peaks(align_mz,tissue_xyz,0.08);
clear k