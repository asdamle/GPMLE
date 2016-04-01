% VERIFY_HUTCHINSON_TRACE Check the trace of the Hutchinson estimator versus the
% true trace
%
%    VERIFY_HUTCHINSON_TRACE(N, N_APPLIES, KERNEL) produces an rskelf factorization of the
%    covariance matrix A (and its derivative A_i) on an N-by-N grid, then uses the Hutchinson estimator to estimate the trace of A^{-1}A_i.
%
%    N               -- int, grid of observations is N by N
%
%    N_APPLIES       -- int, number of random vectors to use in Hutchinson
%                       estimator
%
%    KERNEL          -- char, 'g' for Gaussian (squared-exponential) kernel
%                       and 'm' for Matern
%

function verify_hutchinson_trace(n,n_applies,kernel)
% Assertion may be removed on bigger machines
assert(n <= 64);

if nargin < 2
    n_applies = 100;
end
if nargin < 3
    kernel = 'm';
end

global occ p rank_or_tol noise maxRank x proxy;
occ           = 64;
p             = 16;
rank_or_tol   = 1e-6;
noise         = 1e-2;


% initialize grid
[x1,x2] = ndgrid((1:n)/n);
x = 100*[x1(:) x2(:)]';
x = x.*(1+1e-2*randn(size(x)));
N = size(x,2);


% TODO: Play with this in future
t = hypoct(x,occ);
r = t.nlvl;
maxRank = 100*ones(r,1);

% proxy points are a disc of "nonzero measure"
theta = (1:p)*2*pi/p;
proxy_ = [cos(theta); sin(theta)];
proxy = [];
for r = linspace(1.5,2.5,p)
    proxy = [proxy r*proxy_];
end

theta = [7,10]';

Ffun  = @(x) 0;
Fifun = @(x,i) 0;
% Can choose kernel to test here
if kernel == 'g'
    disp('Using Gaussian (squared-exponential) kernel')
    Ffun    = @(x) gaussian_kernel_2_coord(x);
    Fifun   = @(x,i) gaussian_kernel_grad_2_coord_fast(x,i);
else
    disp('Using Matern family kernel')
    Ffun    = @(x) matern_kernel_2(x);
    Fifun   = @(x,i) matern_kernel_grad_2_fast(x,i);
end
% Consider peeling the first component of the gradient
j   = 1;
[F,A]   = Ffun(theta);
[Fi,Ai] = Fifun(theta,j);


% Approximate trace
if kernel == 'g'
    Avecfun = @(x) rskelf_sv_h(F,rskelf_mv_h_fast(Fi,x));
else
    Avecfun = @(x) rskelf_sv_p(F,rskelf_mv_h_fast(Fi,x));
end

M = n_applies;
R = -1 + 2* (randn(length(x),M) > 0);


approx_trace = trace(R'*Avecfun(R)) / M;

L = A\Ai;
true_trace = trace(L);

disp('The hutchinson trace:')
disp(approx_trace);
disp('The true trace:');
disp(true_trace);
disp('Absolute error hutchinson:');
disp(abs(approx_trace - true_trace));
disp('Relative error hutchinson:');
disp(abs(approx_trace - true_trace) / abs(true_trace));
end