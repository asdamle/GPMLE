% ex1 - a Matern kernel with nu=3/2, true correlation lengths are [7,10]
% points are on a uniform square grid of physical size 100 by 100.


% The true correlation lengths
theta = [7;10];

% Number of points in each direction
n = 32;


global occ p peel_tol rskelf_tol noise maxRank x proxy verb pd;
pd            = 1; % must be 1 here
occ           = 64;
p             = 16;

peel_tol      = 1e-6;
opt_tol = 1e-3;
noise         = 1e-2;

verb          = 0;


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
rskelf_tol_old = rskelf_tol;
rskelf_tol = 1e-12;
F     = matern_kernel_2(theta);
rskelf_tol = rskelf_tol_old;
clear rskelf_tol_old;

f        = rskelf_cholmv(F,g);
%%

funObj = @(theta) likelihood(theta, @matern_kernel_2, @matern_kernel_grad_2_fast, f);


%Evaluate 'true' likelihood
disp('Likelihood of real params');
f = funObj(theta);
format long;
disp(f)
theta_init = [30;3];



% now call the optimization routine


options = optimoptions('fminunc','GradObj','on','display','iter','algorithm','quasi-newton','tolx',1e3*peel_tol,'tolfun',1e3*peel_tol);


clear F;
t=tic();
[theta_opt,fval,exitflag,output]= fminunc(funObj,theta_init,options);
mytime=toc(t)

disp('Final theta_opt is:');
disp(theta_opt);
mywhos = whos;