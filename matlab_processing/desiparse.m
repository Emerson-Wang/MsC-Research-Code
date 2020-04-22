function [mcvec, matxy, matspec] = desiparse(fname)
% [MCVEC, MATXY, MATSPEC]=DESIPARSE(FNAME) parses file FNAME
% to find the mass/charge ratios in MCVEC and a spectral
% image. A pixel's XY coordinates are in MATXY and the
% pixel's spectrum is the corresponding row of MATSPEC

% Open the file and read the mass/charge ratios
fid = fopen(fname, 'r');
line0 = fgetl(fid);
line1 = fgetl(fid);
linendx = fgetl(fid);
mcndx = str2num(linendx);
linespec = fgetl(fid);
rawvec = str2num(linespec);
[foo, mcndx] = sort(rawvec);
mcvec = rawvec(mcndx);

% Loop to create the image
N = numel(mcndx)

matxy = zeros(1, 2);
matspec = zeros(1, N);
ix = 1;
while 1
  thisfline = fgetl(fid);
  if ~ischar(thisfline), break, end
  thisnum = str2num(thisfline);
  if (numel(thisnum)==0), break, end
  thisunsrt = thisnum(4:(end-2));
  matxy(ix, :) = thisnum(2:3);
  matspec(ix, :) = thisunsrt(mcndx);
  ix = ix + 1;
  end

% Close the file and return
fclose(fid);
