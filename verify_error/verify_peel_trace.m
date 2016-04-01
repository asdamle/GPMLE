% VERIFY_PEEL_TRACE Check the trace of the peeling algorithm versus the
% true trace
%
%    VERIFY_PEEL_TRACE(N, PEEL_TOL, WEAK_OR_STRONG, KERNEL, TIME_IT, VERBOSE) produces an rskelf factorization of the
%    covariance matrix A (and its derivative A_i) on an N-by-N grid, then uses the peeling algorithm to estimate the trace of A^{-1}A_i.
%
%    N               -- int, grid of observations is N by N
%
%    PEEL_TOL        -- float, the criterion for low-rank blocks in the peeling
%                       algorithm
%    WEAK_OR_STRONG  -- char, 'w' for weak peeling (HODLR), 's' for strong
%                       peeling (classic regular admissility)
%
%    KERNEL          -- char, 'g' for Gaussian (squared-exponential) kernel
%                       and 'm' for Matern
%
%    TIME_IT         -- bool, 1 to run peeling twice to time peeling
%
%    VERBOSE         -- bool, 1 to enable verbose output



function verify_peel_trace(n,peel_tol, weak_or_strong, kernel, time_it, verbose)
% Assertion may be removed on bigger machines
assert(n <= 64);

if nargin < 2
    peel_tol = 1e-6;
end
if nargin < 3
    weak_or_strong = 'w';
end
if nargin < 4
    kernel = 'm';
end
if nargin < 5
    time_it = 1;
end
if nargin < 6
    verbose = 1;
end

global occ p rskelf_tol noise maxRank x proxy;
occ           = 64;
p             = 16;
rskelf_tol    = peel_tol/1000;
noise         = 1e-2;



% initialize grid
[x1,x2] = ndgrid((1:n)/n);
x = 100*[x1(:) x2(:)]';

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

mat = (A\Ai + Ai/A) / 2;


% Approximate trace
if kernel == 'g'
    % Unfortunately, Gaussian kernel has trouble staying positive
    % definite for small noise --> assume it is Hermitian but maybe
    % indefinite
    Avecfun = @(x) 1/2*(rskelf_sv_h(F,rskelf_mv_h_fast(Fi,x)) + rskelf_mv_h_fast(Fi,rskelf_sv_h(F,x)));
else
    Avecfun = @(x) 1/2*(rskelf_sv_p(F,rskelf_mv_h_fast(Fi,x)) + rskelf_mv_h_fast(Fi,rskelf_sv_p(F,x)));
end

disp('Starting peeling');

if weak_or_strong == 's'
    disp('Using strong peeling');
    [approx_trace, maxRank] = peel_strong(Avecfun,x,occ,peel_tol,maxRank,verbose);
else
    disp('Using weak peeling');
    [approx_trace, maxRank] = peel(Avecfun,x,occ,peel_tol,maxRank,verbose);
end

if time_it
    disp('Running again for timing purposes');
    t = tic();
    if weak_or_strong == 's'
        [approx_trace, maxRank] = peel_strong(Avecfun,x,occ,peel_tol,maxRank,verbose);
    else
        [approx_trace, maxRank] = peel(Avecfun,x,occ,peel_tol,maxRank,verbose);
    end
    toc(t)
end
disp('Peeling complete');


true_trace   = trace(mat);

disp('The peeled trace:')
disp(approx_trace);
disp('The true trace:');
disp(true_trace);
disp('Absolute error:');
disp(abs(approx_trace - true_trace));
disp('Relative error:');
disp(abs(approx_trace - true_trace) / abs(true_trace));


end