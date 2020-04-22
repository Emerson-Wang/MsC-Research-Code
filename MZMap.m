%%% Finds the integer index mapping from one set of M/Z ratios to the
%%% template spectrum. This is done by finding the smallest difference
%%% in M/Z ratio from one spectrum to the other and mapping the mcvec index
%%% to the template index.
%%% INPUT: mcvec: original M/Z spectrum
%%%         template: desired reference spectrum
%%% OUTPUT: mappedSpectrum: spectrum that contains a mapping at each index
%%% to the template spectrum
function [mappedSpectrum] = MZMap(spec, mcvec, template)
    mappedSpectrum = zeros(1, size(mcvec,2));
    newSpec = zeros(1,size(spec,2));
    parfor ix = 1:size(mcvec,2)
        mappedSpectrum(ix) = binarymapiter(mcvec(ix),template);
    end
end