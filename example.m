% This script provides a working example of using the given code to perform 
% maximum likelihood estimation for parameter-fitting given 
% observations from a kernelized Gaussian process in two spatial dimensions.

%% set up some globals used by FLAM and the matrix peeling code
clear
global occ p peel_tol rskelf_tol noise maxRank x proxy verb pd;

%% various parameters consistently used throughout the code
pd            = 1;          % 1 to ensure pos def routines are used 
occ           = 64;         % quadtree occupancy per box at the leaf level
p             = 16;         % number of points on the proxy surface
peel_tol      = 1e-6;       % tol used in the peeling computations for the trace
noise         = 5e-2;       % sqrt of noise factor added to the diagonal
verb          = 0;          % 1 for verbose output
rskelf_tol = peel_tol/1000; % tol used when computing the rskelf factorizations

%% set up spatial observation locations
% first, we set up the spatial points at which measurements are taken. 

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

% x is 2 x N where N is the number of points, each column is a data
% location
N = 2^11;
x = pts(:,sub(1:N));


%% set up some required components for rskelf and peeling

% Give an initial guess of what the ranks should be for peeling,
% but getting this wrong only hurts us the first time due to adaptivity
t = hypoct(x,occ);
r = t.nlvl;
maxRank = 100*ones(r,1);



% set up the proxy points used in the rskelf algorithm
theta_proxy = (1:p)*2*pi/p;
proxy_ = [cos(theta_proxy); sin(theta_proxy)];
proxy = [];
for r = linspace(1.5,2.5,p)
    proxy = [proxy r*proxy_];
end

clear theta_proxy proxy_;


%% generate some "observed" data on the spatial points
rng(0)
g = randn(N,1);

% compute a high accuracy rskelf factorization to synthetically generate data
% from a two parameter Matern kernel (more on kernels later)
theta_true = [7 10];
rskelf_tol_old = rskelf_tol;
rskelf_tol = 1e-14;
F     = matern_kernel_2(theta_true);
rskelf_tol = rskelf_tol_old;

clear rskelf_tol_old;

% generate the data by applying the generalized Cholesky factor to g
f = rskelf_cholmv(F,g);
clear F;



%% set up to solve the MLE problem
% Here we set up the object for the log-likelihood and its derivative.
% The anonymous function is dependent on the kernel parameters theta
% Here, matern_kernel_2 and matern_kernel_grad_2_fast are given as function
% handles to files in ex/ that define the kernel and its derivative
% See ex/matern_kernel_2.m and ex/matern_kernel_grad_2_fast.m for examples
% of how to construct a kernel function and a function for the derivatives
% of the kernel. Lastly, f is the data.


funObj = @(theta) likelihood(theta, @matern_kernel_2, @matern_kernel_grad_2_fast, f);



%% Finally, we will set up and solve the MLE problem
%Evaluate true likelihood using the true parameter values
t = tic();
[f] = funObj(theta_true);
mytime1 = toc(t);
disp('Likelihood at the true parameters');
disp(f);
disp('Time taken to compute:');
disp(mytime1)


% start the optimization problem somewhere else
theta_init = [30;3];



% decide if you want to actually compute the derivatives using the required
% rskelf and peeling routines, or simply use finite differences
fd_or_peel = 'p';

% set up the options for MATLABs built in optimization routine.
s = 'on';
if ~(fd_or_peel == 'p')
    s = 'off';
end
opt_tol = 1e-3;
options = optimoptions('fminunc','GradObj',s,'display','iter','algorithm','quasi-newton','tolx',opt_tol,'tolfun',opt_tol);

% try and maximize the likelihood using fminunc. One may also use fmincon
% with the appropriate options setup if the parameters need to be
% constrained.
t=tic();
[theta_opt,fval,exitflag,output]= fminunc(funObj,theta_init,options);
mytime=toc(t);

disp('True parameter values');
disp(theta_true);

disp('Initial guess for parameters');
disp(theta_init);

disp('Parameters computed via MLE');
disp(theta_opt);

disp('Total time taken by the optimization routine');
disp(mytime);


