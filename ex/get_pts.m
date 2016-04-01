function [pts sst] = get_pts(fname,xbounds,ybounds)
% get all or a subset of the points from fname, if xbounds and ybounds are
% empty then get all points. Otherwise, just between the bounds of the form
% xbounds = [xL xH] and ybounds = [yL yH]. Does not make pts unique at the
% moment.


data = dlmread(fname);
data = data(:,[5,6,8]);

latitude = data(:,1);
longitude = data(:,2);
sst = data(:,3);

bit = longitude > 50;

latitude = latitude(bit);
longitude = longitude(bit);
sst = sst(bit);
mapWidth    = 300;
mapHeight   = 200;


xx = (longitude+180)*(mapWidth/360);
latRad = latitude*pi/180;

mercN = log(tan((pi/4)+(latRad/2)));
yy     = (mapHeight/2)-(mapWidth*mercN/(2*pi));

if isempty(xbounds)
    xH = max(xx);
    xL = min(xx);
else
    xH = xbounds(2);
    xL = xbounds(1);
end
if isempty(ybounds)
    yH = max(yy);
    yL = min(yy);
else
    yH = ybounds(2);
    yL = ybounds(1);
end


idx = (xx <= xH) & (xx >= xL);
idy = (yy <= yH) & (yy >= yL);
index = (idx & idy);
sel = 1:length(xx);
sel = sel(index);
xx = xx(sel);
yy = yy(sel);
sst = sst(sel);
pts = [xx(:)' ; yy(:)'];
