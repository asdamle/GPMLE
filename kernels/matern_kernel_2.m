function [F,A] = matern_kernel_2(params)
% Function to construct a parametrized kernel with p parameters for use with
% the included routines to do MLE of the parameters
%
%  INPUT:
%
%  params: a length p vector of parameter values
%
%  OUTPUT:
%
%  F: (required) A rskelf factorization of the kernel
%  A: (optional) A matrix of kernel entries, for debugging on small
%     problems


% build nice parameter
theta1 = params(1);
theta2 = params(2);

% required global variables
global occ rskelf_tol noise x proxy verb pd;

A = [];

% default to 'p' for positive definite, but allow for the more general 'h'
% factorization to be used.
if ~pd
    opts = struct('symm','h','verb',verb);
else
    opts = struct('symm','p','verb',verb);
end

% construct  a rskelf factorization of the kernel for given parameter
% values
F = rskelf(@Afun,x,occ,rskelf_tol,@pxyfun,opts);

% if the dense matrix is desired, construct it
if nargout > 1
    A = Afun(1:length(x), 1:length(x));
end


% function that implements the kernel for given 2 x N vectors of spatial points
% this is all that has to be changed to use a different kernel
    function K = Kfun(x,y)
        dx = bsxfun(@minus,x(1,:)',y(1,:));
        dy = bsxfun(@minus,x(2,:)',y(2,:));
        dr = sqrt(dx.^2/theta1^2 + dy.^2/theta2^2);
        K  = (1+sqrt(3)*dr).*exp(-sqrt(3)*dr);
    end

% construct the matrix entries based on Kfun
    function A = Afun(i,j)
        A = Kfun(x(:,i),x(:,j));
        [I,J] = ndgrid(i,j);
        idx = I == J;
        A(idx) = A(idx) + noise^2;
    end

% proxy function needed for rskelf factorization, this should not need to
% be updated for different kernels
    function [Kpxy,nbr] = pxyfun(x,slf,nbr,l,ctr)
        pxy = bsxfun(@plus,proxy*l,ctr');
        Kpxy = Kfun(pxy,x(:,slf));
        dx = x(1,nbr) - ctr(1);
        dy = x(2,nbr) - ctr(2);
        dist = sqrt(dx.^2 + dy.^2);
        nbr = nbr(dist/l < 1.5);
    end

end