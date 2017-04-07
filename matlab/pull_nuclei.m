function mask = pull_nuclei(path)

if exist(path, 'file'),
	mask = imread(path);
	mask = mask>0;
else,
	msg = sprintf('Path given %s | Does not point to a file', path);
	error(msg);

end


end
