function [Fi,Ai] = gaussian_kernel_grad_2_fast(params,i)

scale = params(1);front = params(2);
global occ rskelf_tol x proxy verb;
opts = struct('symm','h','verb',verb);

Fi = [];
if i == 1
    Fi = rskelf_mv_only(@Afun_grad1,x,occ,rskelf_tol,@pxyfun_grad1,opts);
elseif i == 2
    Fi = rskelf_mv_only(@Afun_grad2,x,occ,rskelf_tol,@pxyfun_grad2,opts);
end

Ai = [];

if nargout > 1
    if i == 1
        Ai = Afun_grad1(1:length(x),1:length(x));
    elseif i == 2
        Ai = Afun_grad2(1:length(x),1:length(x));
    end

end

  % d/dscale of kernel function
  function K = Kfun_grad1(x,y)
    dx = bsxfun(@minus,x(1,:)',y(1,:));
    dy = bsxfun(@minus,x(2,:)',y(2,:));
    dr = sqrt(dx.^2 + dy.^2);
    K = -dr.^2*scale*front.*exp(-0.5*scale^2*dr.^2);
  end
  function K = Kfun_grad2(x,y)
    dx = bsxfun(@minus,x(1,:)',y(1,:));
    dy = bsxfun(@minus,x(2,:)',y(2,:));
    dr = sqrt(dx.^2 + dy.^2);
    K = exp(-0.5*scale^2*dr.^2);
  end


% matrix entries
    function A = Afun_grad1(i,j)
        A = Kfun_grad1(x(:,i),x(:,j));
    end
    function A = Afun_grad2(i,j)
        A = Kfun_grad2(x(:,i),x(:,j));
    end
    

% grad proxy function
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