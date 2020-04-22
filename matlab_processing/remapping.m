% Remaps a spectrum from one MZ vector to another template MZ vector
function [newSpec] = remapping(spec, mcvec, template)
    mapping = MZMap(spec, mcvec, template);
    newSpec = zeros(1,size(template,2));
    for ix = 1:size(spec,2)
       newSpec(mapping(ix)) = newSpec(mapping(ix)) + spec(ix); 
    end