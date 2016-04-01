function [F,A] = gaussian_kernel_2(params)

scale = params(1);
front = params(2);

global occ rskelf_tol noise x proxy verb;


opts = struct('symm','p','verb', verb);
F = rskelf(@Afun,x,occ,rskelf_tol,@pxyfun,opts);

A = [];

if nargout > 1
    A = Afun(1:length(x), 1:length(x));
end


    function K = Kfun(x,y)
        dx = bsxfun(@minus,x(1,:)',y(1,:));
        dy = bsxfun(@minus,x(2,:)',y(2,:));
        dr = scale*sqrt(dx.^2 + dy.^2);
        K  = front*exp(-0.5*dr.^2);
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