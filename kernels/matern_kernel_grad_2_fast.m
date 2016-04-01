function [Fi,Ai] = matern_kernel_grad_2_fast(params,i)
% Function to construct the derivatives (with respect to the parameters) of 
% a parametrized kernel with p parameters for use with
% the included routines to do MLE
%
%  INPUT:
%  
%  params: a length p vector of parameter values
%  i:      an integer in [1,p] specifying wihch derivative to return
%
%  OUTPUT:
%  
%  F: (required) A specialized rskelf factorization of the derivative of the 
%     kernel with respect to parameter i
%  A: (optional) A dense matrix of of the derivative of the 
%     kernel with respect to parameter i

theta1 = params(1);
theta2 = params(2);

%global variables needed
global occ rskelf_tol x proxy verb;

% Construct with the 'h', i.e. use LDL^T rather than Cholesky
opts = struct('symm','h','verb',verb);

% Build a specialized rskelf factorization that may only be used to perform
% matrix vector products with the corresponding specialized routines.
% Return the appropriate one given the input i
Fi = [];
if i == 1
    Fi = rskelf_mv_only(@Afun_grad1,x,occ,rskelf_tol,@pxyfun_grad1,opts);
elseif i == 2
    Fi = rskelf_mv_only(@Afun_grad2,x,occ,rskelf_tol,@pxyfun_grad2,opts);
end


% If desired, build the corresponding dense matrix
Ai = [];

if nargout > 1
    if i == 1
        Ai = Afun_grad1(1:length(x),1:length(x));
    elseif i == 2
        Ai = Afun_grad2(1:length(x),1:length(x));
    end

end

  % derivatives of the kernel function with respect to each parameter,
  % input is two sets of points encoded in 2 x N matrices. This is the only
  % function that needs to be changed to construct a new kernel. Need to
  % have one Kfun_gradi per parameter.
  
  function K = Kfun_grad1(x,y)
    dx = bsxfun(@minus,x(1,:)',y(1,:));
    dy = bsxfun(@minus,x(2,:)',y(2,:));
    dr = sqrt(dx.^2/theta1^2 + dy.^2/theta2^2);
    %dr = sqrt(dx.^2*theta1^2 + dy.^2*theta2^2);
    K = 3*exp(-sqrt(3)*dr).*dx.^2/theta1^3;
    %K = -3*exp(-sqrt(3)*dr).*dx.^2 * theta1;
  end
  function K = Kfun_grad2(x,y)
    dx = bsxfun(@minus,x(1,:)',y(1,:));
    dy = bsxfun(@minus,x(2,:)',y(2,:));
    dr = sqrt(dx.^2/theta1^2 + dy.^2/theta2^2);
    %dr = sqrt(dx.^2*theta1^2 + dy.^2*theta2^2);
    K  = 3*exp(-sqrt(3)*dr).*dy.^2/theta2^3;
    %K = -3*exp(-sqrt(3)*dr).*dy.^2 * theta2;
  end


% compute the matrix entries using Kfun_gradi, need one function per
% parameter
    function A = Afun_grad1(i,j)
        A = Kfun_grad1(x(:,i),x(:,j));
    end
    function A = Afun_grad2(i,j)
        A = Kfun_grad2(x(:,i),x(:,j));
    end

% grad proxy function for Kfun_gradi, need to be one function per parameter
    function [Kpxy,nbr] = pxyfun_grad1(x,slf,nbr,l,ctr)
        pxy = bsxfun(@plus,proxy*l,ctr');
        Kpxy = Kfun_grad1(pxy,x(:,slf));
        dx = x(1,nbr) - ctr(1);
        dy = x(2,nbr) - ctr(2);
        dist = sqrt(dx.^2 + dy.^2);
        nbr = nbr(dist/l < 1.5);
    end

    function [Kpxy,nbr] = pxyfun_grad2(x,slf,nbr,l,ctr)
        pxy = bsxfun(@plus,proxy*l,ctr');
        Kpxy = Kfun_grad2(pxy,x(:,slf));
        dx = x(1,nbr) - ctr(1);
        dy = x(2,nbr) - ctr(2);
        dist = sqrt(dx.^2 + dy.^2);
        nbr = nbr(dist/l < 1.5);
    end

end