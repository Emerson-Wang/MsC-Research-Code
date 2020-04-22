function xyzmat = spec2xyz(matxy, matspec)
% XYZMAT=SPEC2XYZ(MATXY, MATSPEC) creates a 3D matrix
% of DESI mass-spec values; this needs a mass/charge
% vector for full interpretation.

% Sort the XY indexes and find the problem size
vecx = unique(matxy(:,1));
vecy = unique(matxy(:,2));

M = numel(vecx);
N = numel(vecy);
[K L] = size(matspec);

ndx1toM = (1:M)';
ndx1toN = (1:N)';
ndxNto1 = (N:-1:1)';

xyzmat = zeros(M, N, L);
for ix = 1:K
  thisJ = ndx1toM(matxy(ix, 1) == vecx);
  thisI = ndx1toN(matxy(ix, 2) == vecy);
  xyzmat(thisI, thisJ, :) = matspec(ix, :);
end
