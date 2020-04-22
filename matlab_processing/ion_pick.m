function [final_xyz] = ion_pick(mz_vec, batch_xyz, ions)
final_xyz = {1:size(batch_xyz,2)};
for ix = 1:size(batch_xyz,2)
    current_xyz = cell2mat(batch_xyz(ix));
    [m, ~, ~] = size(current_xyz);
    image_ions = [];
    for jx = 1:m
        ion_selected = current_xyz(jx,:);
        ion_idx = ismember(mz_vec,ions);
        ion_selected = ion_selected(ion_idx);
        image_ions = [image_ions; ion_selected];
    end
    final_xyz{ix} = image_ions;
end