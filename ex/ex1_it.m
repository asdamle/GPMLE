% ex1 - a Matern kernel with nu=3/2, true correlation lengths are [7,10]
% points are on a uniform square grid of physical size 100 by 100.

% The true correlation lengths
theta = [7;10];

% Number of points in each direction

n =512;

global occ p peel_tol rskelf_tol noise maxRank x proxy verb pd;
pd            = 1;
occ           = 64;
p             = 16;

peel_tol      = 1e-6;
noise = 1e-2;
verb          = 1;
rskelf_tol = peel_tol/1000;




% initialize grid
[x1,x2] = ndgrid((1:n)/n);
x = [100*x1(:) 100*x2(:)]';
N = size(x,2);

% Gaussian variables to generate data -- do this right after RNG seeding
rng(0);
g      = randn(N,1);


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

f = g;

funObj = @(theta) likelihood(theta, @matern_kernel_2, @matern_kernel_grad_2_fast, f);



%Evaluate true likelihood
t = tic();
[f,g] = funObj(theta);
mytime1 = toc(t)

t=tic();
[f,g] = funObj(theta);
mytime2=toc(t)
