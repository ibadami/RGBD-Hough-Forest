dir =pwd;
filename = fullfile(dir, 'neg.txt');
filename_new = fullfile(dir, 'neg_new.txt');
fd = fopen(filename,'r');
fd_new = fopen(filename_new,'w');
line = fgetl(fd);
[num, t] = strread(line);
fprintf(fd_new,'%d %d\n',num,t);
for i = 1:num,
    line = fgetl(fd);
    image = imread(fullfile('/home/ishrat/Dropbox',['/neg/' line]));
    width = size(image,2);
    height = size(image,1);
    
    newline = ['/neg/' line];
    fprintf(fd_new,[newline ' %d %d %d %d %d %d\n'], 0,0,width,height,round(width/2),round(height/2));
end
fclose(fd);
fclose(fd_new);