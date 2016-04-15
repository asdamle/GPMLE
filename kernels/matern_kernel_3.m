function [F,A] = matern_kernel_3(params)

theta1 = params(1);
theta2 = params(2);
theta3 = params(3);

global occ rskelf_tol noise x proxy verb pd;

A = [];

% TODO: POSITIVE
if ~pd
    opts = struct('symm','h','verb',verb);
else
    opts = struct('symm','p','verb',verb);
end
F = rskelf(@Afun,x,occ,rskelf_tol,@pxyfun,opts);

if nargout > 1
    A = Afun(1:length(x), 1:length(x));
end

    function K = Kfun(x,y)
        dx = bsxfun(@minus,x(1,:)',y(1,:));
        dy = bsxfun(@minus,x(2,:)',y(2,:));
        dr = sqrt(dx.^2/theta1^2 + dy.^2/theta2^2);
        K  = theta3*(1+sqrt(3)*dr).*exp(-sqrt(3)*dr);
    end

% matrix entries
    function A = Afun(i,j)
        A = Kfun(x(:,i),x(:,j));
        [I,J] = ndgrid(i,j);
        idx = I == J;
        A(idx) = A(idx) + noise^2;
    end

% proxy function
    function [Kpxy,nbr] = pxyfun(x,slf,nbr,l,ctr)
        pxy = bsxfun(@plus,proxy*l,ctr');
        Kpxy = Kfun(pxy,x(:,slf));
        dx = x(1,nbr) - ctr(1);
        dy = x(2,nbr) - ctr(2);
        dist = sqrt(dx.^2 + dy.^2);
        nbr = nbr(dist/l < 1.5);
    end

end