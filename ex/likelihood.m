function [f,g] = likelihood(theta, Ffun, Fifun, z)
% Computes -1 times the log-likelihood function and (optionally) its gradient
%
% INPUT:
%
%   theta: a p length vector of the parameters in the kernel
%   Ffun:  an anonymous function specifying the kernel
%   Fifun: an anonymous function specifying the derivatives of the kernel
%   z:     a N x 1 vector containing the data
%
% OUTPUT:
%
%   f:     scalar, the value of the likelihood at theta
%   g:     the p components of the gradient of the likelihood at theta

global x occ peel_tol maxRank verb pd;

persistent rankGuess;
p = length(theta);
if any(size(rankGuess) ~= [length(maxRank),p]);
    rankGuess = zeros(length(maxRank),p);
    for j = 1:p
        rankGuess(:,j) = maxRank;
    end
end

F = Ffun(theta);

% Compute objective function (up to affine transformation)
if pd
    Finvz   = rskelf_sv_p(F,z);
else
    Finvz   = rskelf_sv_h(F,z);
end

term1 = -z'*Finvz;
term2 = -rskelf_logdet(F);

f =  term1 + term2;
% we actually want the negative log-likelihood because we minimize
f = -f;

num_grad = p;
% Compute gradient
if nargout >= 2
    g = zeros(num_grad,1);
    for j = 1:num_grad
        Fi = Fifun(theta,j);
        term_1 = Finvz'*rskelf_mv_h_fast(Fi,Finvz);
        % peel
        Avecfun = @(x) 0;
        if ~pd
            Avecfun = @(x) 0.5*(rskelf_sv_h(F,rskelf_mv_h_fast(Fi,x)) + rskelf_mv_h_fast(Fi, rskelf_sv_h(F,x)));
        else
            Avecfun = @(x) rskelf_cholsv(F,rskelf_mv_h_fast(Fi,rskelf_cholsv(F,x,'c')));
        end
        [tr,rankGuess(:,j)] = peel(Avecfun,x,occ,peel_tol,rankGuess(:,j),verb);
        term_2 =  -tr;
        g(j) = term_1 + term_2;
        
    end
    % we actually want the negative log-likelihood because we minimize
    g = -g;
end

end
