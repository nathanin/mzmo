pkg load image
pkg load statistics
pkg load communications

mask0 = '/home/nathan/mzmo/data/segmented.1/m0/labels';
mask1 = '/home/nathan/mzmo/data/segmented.1/m1/labels';

root0 = '/home/nathan/mzmo/data/source_nuclei_0';
root1 = '/home/nathan/mzmo/data/source_nuclei_1';

write0 = '/home/nathan/mzmo/data/nuclei/indiv.interior.2/0';
write1 = '/home/nathan/mzmo/data/nuclei/indiv.interior.2/1';

% profile on;
maskfns(root0, write0, mask0, 'indiv_interior - area > 442 ecc < 0.8');
% profile off;

% data = profile('info');
% profshow(data, 10);

% profile on;
maskfns(root1, write1, mask1, 'indiv_interior - area > 442 ecc < 0.8');
% profile off;

% data = profile('info');
% profshow(data, 10);