%

function varargout = maskfns(varargin)
	%rootdir = varargin{1};
	%writep = varargin{2};
	%glmaskpath = varargin{3};
	%versionStr = varargin{4};

    rootdir = '/home/ingn/mzmo/data/nuclei/source_nuclei_1';
    writep = '/home/ingn/mzmo/data/nuclei/indiv.fixed64.pick20.nodilate/1';
    writeLL = '/home/ingn/mzmo/data/nuclei/indiv.fixed64.pick20.nodilate/LL1';
    glmaskpath = '/home/ingn/mzmo/data/';
    versionStr = 'fixed 64 window - hard thresholds - pick 20'; 
   
    extract_template(rootdir, writep, writeLL, glmaskpath, versionStr);

    %$fprintf('Rootdir: %s\n', rootdir)
    %$fprintf('writep: %s\n', writep)
    %$fprintf('glmaskpath: %s\n', glmaskpath)
    %$fprintf('versionStr: %s\n', versionStr)

	%$if ~exist(writep, 'dir'),
	%$	mkdir(writep)
	%$end

	%$if ~exist(writeLL, 'dir'),
	%$	mkdir(writeLL)
	%$end

	%$%versiondoc = make_version_notes(writep);

	%$%if exist(versiondoc),
	%$%	versionFileID = fopen(versiondoc, 'a');
	%$%	fprintf(versionFileID, '\n\n');
	%$%else
	%$%	versionFileID = fopen(versiondoc, 'w');
	%$%end

% TO%$DO move this outsize/create programmatically. 
	%$% versionStr = 'indiv.4';
	%$note_txt = 'Gray img, unmasked';
	%$writeout = true;
	%$listpath = make_listfile(writep);
	%$listfileID = fopen(listpath, 'w');

    %$indicespath = make_indexfile(writep);
    %$indicesID = fopen(indicespath, 'w');

	%$subdirs = {'mask', 'feature'};
	%$[masklist, featlist] = get_lists(rootdir, subdirs);

	%$n_imgs = length(masklist);

	%$for k=1:n_imgs,
	%$	if mod(k, 50) == 0
	%$		fprintf('%g / %g\n', k, n_imgs);
	%$	end
	%$	% fprintf('%g', k)
	%$	run_img(masklist{k}, featlist{k}, ...
	%$				     writep, glmaskpath, ...
	%$				     listfileID, writeout, indicesID, writeLL);
	%$end

	%$fclose(listfileID);
	%$fclose(indicesID);
    %$quit()
end


%%
%function printout(stats, n_imgs, listpath, rootdir, note, versionFileID, versionStr)
%
%	fprintf(versionFileID, '>>>>>>>>>>>>>>>>>---------Version --------| %s |------\n', versionStr);
%	fprintf(versionFileID, 'Root dir %s\n', rootdir);
%	fprintf(versionFileID, 'Note: %s\n', note);
%	fprintf(versionFileID, '\tImage list: %s\n', listpath);
%	fprintf(versionFileID, '\tNumber of images: \t%g\n', n_imgs);
%
%	fields = fieldnames(stats);
%	for k = 1:length(fields)
%		f = fields{k};
%		fprintf(versionFileID, '\t%s: \t%3.3f\n', f, stats.(f)/n_imgs);
%	end
%end
%
%
%%
%function combo = track_stats(combo, statout)
%	fields = fieldnames(combo);
%
%	for k = 1:length(fields)
%		combo.(fields{k}) = combo.(fields{k}) + statout.(fields{k});
%	end
%end
%
%
%%
%function verpath = make_version_notes(writep)
%	[wd, n, ~] = fileparts(writep);
%	verpath = fullfile(wd, sprintf('version-notes-small.txt', n));
%	return;
%end
%
%
%
%%
%function lfpath = make_listfile(writep)
%	% infer the place for listfile to go from writep
%	[wd, n, ~] = fileparts(writep);
%	fprintf('Creating listfile at %s\n', writep)
%	lfpath = fullfile(writep, sprintf('%s.txt', n)); 
%
%	return;
%end
%
%
%%
%function lfpath = make_indexfile(writep)
%	% infer the place for listfile to go from writep
%	[wd, n, ~] = fileparts(writep);
%	fprintf('Creating indexfile at %s\n', writep)
%	lfpath = fullfile(writep, sprintf('indices_%s.txt', n)); 
%
%	return;
%end
%
%
%%
%function varargout = get_lists(root, subdirs)
%	for k=1:length(subdirs),
%		f = dir(fullfile(root, subdirs{k}, '*.png'));
%		f = {f.name};
%		varargout{k} = fullfile(root, subdirs{k}, f);
%	end
%end
%
%
%% 
%function mask = get_mask(pth, mmode)
%	if exist(pth, 'file'),
%		mask = imread(pth);
%
%		% TODO magic numbers
%		if strcmp(mmode, 'nuclei')
%			mask = mask>0;
%		elseif strcmp(mmode, 'gland')
%			mask = mask ~= 3;
%		end
%
%		return
%	else,
%		msg = sprintf('Path given %s | Does not point to a file\n', pth);
%		error(msg);
%	end
%end
%
%
%%
%function img = get_img(pth)
%	if exist(pth, 'file'),
%		img = imread(pth);
%
%		return
%	else,
%		msg = sprintf('Path given %s | Does not point to a file\n', pth);
%		error(msg);
%	end
%end
%
%
%% colors = [0,0,0];
%function overlay = overlay_mask(mask, img)
%	colors = [0,0,0];
%	maskdims = size(mask);
%	imgdims = size(img);
%	if all(maskdims== imgdims(1:2)),
%		mask = bwmorph(mask, 'remove');
%
%		for k=1:3
%			x = img(:,:,k);
%			x(mask) = colors(k);
%			img(:,:,k) = x;
%		end
%
%		overlay = img;
%		return
%
%	else,
%		msg = sprintf('Mask and img sizes disagree')
%		error(msg)
%	end
%end
%
%
%%
%function varargout = connected_objects(mask, statlist, varargin)
%	CC = bwconncomp(mask);
%	numPixels = cellfun(@numel,CC.PixelIdxList);
%	% objmean = mean([numPixels]);
%	% objstd = std([numPixels]);
%
%	% fprintf('\tMean area: %g std: %g\n', objmean, objstd)
%	if nargin == 2
%		stats = regionprops(CC, statlist);
%	elseif nargin == 3
%		I = varargin{1};
%		stats = regionprops(mask, I, statlist);
%	end
%
%    for k = 1:length(stats),
%        stats(k).index = k;
%    end
%
%	varargout{1} = CC;
%	varargout{2} = stats;
%
%	if nargout == 3
%		varargout{3} = bwlabel(mask);
%	end
%
%	return
%end
%
%
%%
%function [x,y] = pullC(c)
%	x = c(1);
%	y = c(2);
%end
%
%
%%
%function stats = remove_edge_nuclei(LL, stats, wind)
%	%dims = CC.ImageSize;
%    dims = size(LL);
%	h = dims(1); w = dims(2);
%
%	hwind = floor(wind/2);
%
%	% left = hwind;
%	% right = w-hwind;
%	% top = hwind;
%	% bot = h-hwind;
%
%	passed = false(1,length(stats));
%	for k = 1:length(stats),
%		[x,y] = pullC(stats(k).Centroid);
%		if (x-hwind)>=1 && (x+hwind)<w &&...
%		   (y-hwind)>=1 && (y+hwind)<h,
%			passed(k) = true;
%		end
%	end
%
%	stats = stats(passed);
%
%	return
%end
%
%
%%
%function objects = get_objects(img, stats, wind)
%	hwind = ceil(wind/2);
%
%	objects = cell(length(stats),1);
%	for k = 1:length(stats),
%		[x,y] = pullC(stats(k).Centroid);
%		bbox = [x-hwind, y-hwind, wind-1, wind-1];
%
%		objects{k} = imcrop(img, bbox);
%	end
%
%
%	return;
%end
%
%
%function img = create_masked_object(img, mask)
%	%TODO Implement hole filling and edge smoothing. 
%	%mask = imdilate(mask, se);
%	r = img(:,:,1); g = img(:,:,2); b = img(:,:,3);
%	r(~mask) = 0;
%	g(~mask) = 0;
%	b(~mask) = 0;
%
%	img = cat(3, r,g,b);
%
%	return;
%end
%
%
%%
%function writeobjects(imgp, writep, objects)
%	[~, name, ~] = fileparts(imgp);
%
%	fprintf('\tWriting to %s .... ', writep)
%	for k = 1:length(objects)
%		name_ = fullfile(writep, sprintf('%s_%g.png', name, k));
%
%		imwrite(objects{k}, name_);
%	end
%	fprintf('Done\n')
%end
%
%
%function imout = get_overlapping_nuclei(LL, mask)
%	imout = LL;
%	if any(size(mask) ~= size(imout))
%		% who mask
%		% class(mask)
%		mask = imresize(uint8(mask), size(imout));
%	end
%
%	imout(mask == 0) = 0;
%
%	return;
%end
%
%
%function maskout = load_one_mask(maskp)
%    maskout = get_mask(maskp, 'nuclei');
%    %maskout = imdilate(maskout, strel('disk', 1, 0));
%end
%
%
%function [nlmask, glmask, overlapping] = load_both_masks(maskp, gpath)
%	nlmask = get_mask(maskp, 'nuclei'); 
%	%glmask = get_mask(gpath, 'gland');
%
%	%glmask = imopen(glmask, strel('disk', 3, 0));
%
%	% nlmask = bwfill(nlmask, 'holes');
%	%nlmask = imdilate(nlmask, strel('disk', 1, 0));
%	%overlapping = get_overlapping_nuclei(nlmask, glmask);
%    
%   % Temp fix while i wait for good gl masks:
%    glmask = 0;
%    overlapping = nlmask; 
%end
%
%
%function [img_masked, img_gray, img_masked_gray] = process_image(img, overlapping)
%	img_masked = create_masked_object(img, overlapping);
%	img_gray = Color_DeconvolutionPC(img);
%	img_gray = img_gray(:,:,1);
%	img_gray = imcomplement(img_gray);
%
%	img_masked_gray = img_gray;
%	[m,n] = size(img_gray);
%	gaussnoise = get_gaussian_noise(m,n);
%	img_masked_gray(~overlapping) = gaussnoise(~overlapping);
%end
%
%
%function write_LL(LL, fb, writep)
%    writeout = fullfile(writep, sprintf('%s_LL.png', fb))
%    imwrite(LL, writeout)
%end
%
%
%function write_indices(stats, fb, indicesID)
%% Print out one line listing the captured nuclei indices
%    survivors = [stats.index];
%    
%    fprintf(indicesID, '%s',  fb);
%    for k = 1:length(survivors),
%        fprintf(indicesID, ',%g', survivors(k));
%    end
%
%    fprintf(indicesID, '\n');
%
%end
%
%
%function write_objects(objects, writep, fb, listfileID, wind_write)
%	for k=1:length(objects)
%		obj = imresize(objects{k}, [wind_write, wind_write], 'bicubic');
%		obj_n = sprintf('%s_%g.jpg', fb, k);
%		obj_n = fullfile(writep, obj_n);
%
%		fprintf(listfileID, '%s\n', obj_n);
%		imwrite(obj, obj_n);
%	end
%end
%
%
%function stats = filter_area(stats, thresh)
%	areas = [stats.Area];
%	passing = areas > thresh;
%
%	stats = stats(passing);
%end
%
%function stats = filter_eccen(stats, thresh)
%	ecc = [stats.Eccentricity];
%	passing = ecc <= thresh;
%
%	stats = stats(passing);
%end
%
%function gaussnoise = get_gaussian_noise(m,n)
%	gaussnoise = abs( randn(m,n,1) );
%	% Scale
%	high = max(max(gaussnoise));
%	fact = floor(255/high);
%
%	gaussnoise = gaussnoise.*fact;
%	gaussnoise(gaussnoise > 255) = 255;
%end
%
%
%function stats = filter_percent(stats, pct)
%    stats = randsample(stats, round(length(stats)*pct));
%    
%end
%
%
%function stats = filter_n(stats, n)
%    if length(stats) <= n,
%        return 
%    end
%    stats = randsample(stats, n);
%end
%
%
%%
%function run_img(maskp, featp, writep, glmaskp, listfileID, writeout, indicesID, writeLL)
%	% This stuff just exists to comepensate for bad data management
%	[~,mb,~] = fileparts(maskp);
%	[~,fb,~] = fileparts(featp);
%	%gpath = fullfile(glmaskp, [fb, '.png']);
%	%if ~strcmp(mb, fb) || ~exist(gpath),
%	%	error('Paths disagree')
%	%end
%    
%    if ~strcmp(mb, fb),
%        error('Paths disagree')
%    end
%        
%	wind = 64;
%	wind_write = 256;
%	% Load masks
%	%[nlmask, glmask, overlapping] = load_both_masks(maskp, gpath);
%	overlapping = load_one_mask(maskp);
%    
%	img = get_img(featp); 
%	%ovlp = imdilate(overlapping, strel('disk', 1, 0));
%	[img_masked, img_gray, img_masked_gray] = process_image(img, overlapping);
%
%	% Stats
%	statlist = {'Area', 'Centroid', 'Eccentricity'};
%	% overlapping = imdilate(overlapping, strel('disk', 1, 0));
%	[CC, stats, LL] = connected_objects(overlapping, statlist, img_gray);
%	n_overlapping_with_edge = length(stats);
%
%	stats = remove_edge_nuclei(LL, stats, wind);	
%	stats = filter_area(stats, 442);
%	stats = filter_eccen(stats, 0.8);
%
%    %stats = filter_percent(stats, 0.05);
%    stats = filter_n(stats, 10);
%
%	% Write out the remaining
%	if writeout,
%		objects = get_objects(img_gray, stats, wind);
%		write_objects(objects, writep, fb, listfileID, wind_write);
%
%        write_indices(stats, fb, indicesID);
%
%        write_LL(LL, fb, writeLL)
%	end
%
%	%n_original = length(unique(unique(bwlabel(nlmask))));
%	%n_overlapping = length(stats);
%
%	%% Mean area
%	%areas = [stats.Area];
%	%mean_area = mean(areas);
%	%std_area = std(areas);
%
%	%% Mean intensities
%	%intensities = [stats.MeanIntensity];
%	%mean_intensity = mean(intensities);
%	%std_intensity = std(intensities);
%
%	%% Mean eccentricities
%	%eccentricities = [stats.Eccentricity];
%	%mean_eccentricity = mean(eccentricities);
%	%std_eccentricity = std(eccentricities);
%
%	%% Gland percentage
%	%%gland_pct = nnz(glmask)/numel(glmask);
%	%%nuclei_pct = n_overlapping_with_edge/n_original;
%    %gland_pct = 0;
%    %nuclei_pct = 0;
%
%	%varargout{1} = struct('n_nuclei', n_overlapping, 'mean_area', mean_area, ...
%	%					'mean_intens', mean_intensity, 'mean_eccen', mean_eccentricity,...
%	%					'gland_pct', gland_pct, 'nuclei_pct', nuclei_pct);
%
%	%fprintf('%s: N:%g\tArea:%g\tIntens:%g\tEccen:%g\tGLpct:%g\tNuclPct:%g\n',...
%	%		fb, n_overlapping, mean_area, mean_intensity, mean_eccentricity, gland_pct, nuclei_pct);
%	return;
%end
%
%
%
%
