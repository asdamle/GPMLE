%    References:
%
%      Lin Lin, Jianfeng Lu, and Lexing Ying. 2011. Fast construction of
%      hierarchical matrix representation from matrix-vector multiplication.
%      J. Comput. Phys. 230, 10 (May 2011), 4071-4087.

function [F] = apply_partial_peel(GHat,T,Y,x,L)

N = size(x,2);

m = size(Y,2);

% Initialize with zeros before we tack things on
F = zeros(N,m);

for iPrev = 2:L
    n_nodes = T.lvp(iPrev+1) - T.lvp(iPrev);
    
    % Compute transfer
    for I = 1:n_nodes
        iNodeIdx = T.lvp(iPrev) + I;
        sibl = T.nodes(iNodeIdx).sibl;
        sibl = sibl - T.lvp(iPrev);
        for J = sibl
            if I < J
                M_IJ = GHat{iPrev}(I,J).M;
            else
                M_IJ = GHat{iPrev}(J,I).M';
            end
            jNodeIdx = T.lvp(iPrev) + J;
            xj       = T.nodes(jNodeIdx).xi;
            xi = T.nodes(iNodeIdx).xi;
            F(xi,:) = F(xi,:) + GHat{iPrev}(I,J).U*(M_IJ * (GHat{iPrev}(J,I).U'*Y(xj,:)));
        end
    end
    
    
end

end