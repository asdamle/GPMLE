% ex2 - a Matern kernel with nu=3/2, true correlation lengths are [7,10]
% points are non-uniform from the sea surface data, scaled to a [0,100]^2
% domain, this is a situation where finite differences fail and peeling
% works


% The true correlation lengths
theta = [7;10];

% Number of total points

n = 11;
N  = 2^n;

% p for peeling, anything else uses finite differences
fd_or_peel = 'p';

global occ p peel_tol rskelf_tol noise maxRank x proxy verb pd;
pd            = 1;
occ           = 64;
p             = 16;
peel_tol      = 1e-6;
opt_tol = 1e-3;
noise = 5e-2;
verb          = 0;

rskelf_tol = peel_tol/1000;
if ~(fd_or_peel == 'p')
    rskelf_tol = 1e-15;
end




% Gaussian variables to generate data -- do this right after RNG seeding
rng(0);
g      = randn(N,1);

% Use ocean data from
% National Climatic Data Center/NESDIS/NOAA/U.S. Department of Commerce, Data Support Section/
% Computational and Information Systems Laboratory/National Center for Atmospheric Research/
% University Corporation for Atmospheric Research, Earth System Research Laboratory/NOAA/
% U.S. Department of Commerce, and Cooperative Institute for Research in Environmental Sciences/
% University of Colorado (1984):
% International Comprehensive Ocean-Atmosphere Data Set (ICOADS) Release 2.5, Individual Observations.
% Research Data Archive at the National Center for Atmospheric Research, Computational and Information Systems Laboratory.
% Dataset. http://dx.doi.org/10.5065/D6H70CSV. Accessed 11 Nov 2015.

[pts, sst] = get_pts('ocean_data.txt',[],[]);

max_x = max(pts(1,:));
min_x = min(pts(1,:));
max_y = max(pts(2,:));
min_y = min(pts(2,:));
pts(1,:) = (pts(1,:) - min_x)*100/(max_x-min_x);
pts(2,:) = (pts(2,:) - min_y)*100/(max_y-min_y);

Ntotal = length(sst);
sub = randperm(Ntotal);
x = pts(:,sub(1:N));


% Give an initial guess of what the ranks should be for peeling,
% but this only hurts us the first time due to adaptivity
t = hypoct(x,occ);
r = t.nlvl;
maxRank = 100*ones(r,1);



% proxy points are a disc of "nonzero measure" for numerical analytic
% continuation (BAD)
theta_proxy = (1:p)*2*pi/p;
proxy_ = [cos(theta_proxy); sin(theta_proxy)];
proxy = [];
for r = linspace(1.5,2.5,p)
    proxy = [proxy r*proxy_];
end

clear theta_proxy proxy_;



% Generate true observations
rskelf_tol_old = rskelf_tol;
rskelf_tol = 1e-12;
F     = matern_kernel_2(theta);
rskelf_tol = rskelf_tol_old;

clear rskelf_tol_old;

f        = rskelf_cholmv(F,g);
clear F;


funObj = @(theta) likelihood(theta, @matern_kernel_2, @matern_kernel_grad_2_fast, f);




%Evaluate true likelihood
t = tic();
[f] = funObj(theta)
mytime1 = toc(t)



theta_init = [30;3];



% now call the optimization routine
s = 'on';
if ~(fd_or_peel == 'p')
    s = 'off';
end
options = optimoptions('fminunc','GradObj',s,'display','iter','algorithm','quasi-newton','tolx',opt_tol,'tolfun',opt_tol);


t=tic();

[theta_opt,fval,exitflag,output]= fminunc(funObj,theta_init,options);
mytime=toc(t)

disp('Final theta_opt is:');
disp(theta_opt);
