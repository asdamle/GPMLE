% PEEL Basic matrix peeling algorithm of Lin et al.
%
%    [TRACE_EST,MAXRANKSOBSERVED] = PEEL(A, X, OCC, TOL, MAXRANK, VERB) produces a factorization F of the
%    interaction matrix A on the points X using tree occupancy parameter OCC,
%    local precision parameter TOL, maximum rank at each level MAXRANK and
%    verbosity parameter VERB.
%
%    Afun      -- the N by N fast operator to approximate, should be apply-able
%                 via B = Afun(Y) to a matrix Y
%    X         -- the point set, d by N
%    OCC       -- stop when diagonal blocks have no more than OCC points
%    TOL       -- the tolerance for the SVD in order to determine rank
%    MAXRANK   -- the maximum allowable rank for an off-diagonal block at
%                 level 2 to L.  This is a vector
%    VERB      -- bool, 1 for verbose output
%
%    TRACE_EST        -- the estimate of the trace of Afun as an operator
%    MAXRANKSOBSERVED -- a vector of ranks observed at each level that is
%                        useful for expediting multiple calls to similar matrices
%                        (optional)
%
%    References:
%
%      Lin Lin, Jianfeng Lu, and Lexing Ying. 2011. Fast construction of 
%      hierarchical matrix representation from matrix-vector multiplication. 
%      J. Comput. Phys. 230, 10 (May 2011), 4071-4087.


function [trace_est,varargout] = peel(Afun, x, occ, tol, maxRank, verb)


nargoutchk(1,2);

% Set BLOCK_SIZE to choose the maximum number of random vectors
% to apply Afun to at one time for efficiency reasons.  (seems to have
% little effect)
BLOCK_SIZE = 1e6;

% set default parameters
if nargin < 4
    tol = 1e-6;
end
if nargin < 6
    verb = 1;
end

d = size(x,1);
N = size(x,2);

% At each level, split a box into nSplit boxes
nSplit = 2^d;

% Build a top-down tree on the points to find index sets later
tic;
T = hypoct_td(x, occ);
if verb
    v_total = 0;
    t_total = 0;
    fprintf(['-'*ones(1,80) '\n']);
    fprintf('tree\n');
    fprintf(['-'*ones(1,80) '\n']);
    temp = toc;
    t_total = t_total + temp;
    fprintf('%3s | %22.2e (s)\n','-',temp);
    fprintf(['-'*ones(1,80) '\n']);
    fprintf(['lvl |  r (min, max)  | t\n']);
    fprintf(['-'*ones(1,80) '\n']);
end

% Note: for asymptotic efficiency, nLevel should be O(log N)
nLevel = T.nlvl;
if nargin < 5 || length(maxRank) ~= nLevel
    maxRank = 20*ones(nLevel,1);
end


maxRanksObserved = zeros(nLevel,1);


% We require a total of N by nSample random numbers per level
% We will index into sub-blocks accordingly to break up into R_{1;1} etc.
overSample = 10;

nSample = maxRank + overSample;
R = cell(nLevel,1);
for i = 1:nLevel
    R{i} = randn(N,nSample(i));
end

GHat = cell(nLevel,1);
for iLevel = 2:nLevel
    n_nodes = T.lvp(iLevel+1) - T.lvp(iLevel);
    % Note: always store information for tuple (I,J) at
    % the tuple (min(I,J), max(I,J))
    GHat{iLevel} = repmat(struct('U', [], 'M', []),n_nodes,n_nodes);
end

redo_flag = 0;

% Perform the peeling for each level
iLevel = 2;
while iLevel <= nLevel
    % info for verbose output
    if ~redo_flag
        lvlStart = tic();
    end
    
    r_avg              = 0;
    r_min              = inf;
    r_max              = 0;
    n_r_computed       = 0;
    
    n_nodes = T.lvp(iLevel+1) - T.lvp(iLevel);
    
    % GR is product of G with random matrices R
    % Store this as a square cell array but do not use the
    % diagonal
    if ~redo_flag
        GR = cell(n_nodes);
    end
    
    for jChild = 1:nSplit
        % Construct the probing matrix Y, which has a sparsity pattern where only
        % blocks of Y corresponding to sources in the jChild-th child of a
        % node on level iLevel-1 is nonzero
        
        if ~redo_flag
            Y = zeros(N,nSample(iLevel));
        else
            Y = zeros(N,nSample(iLevel)/2);
        end
        
        % The number of blocks is the number of nodes on the previous
        % level
        nblocks = T.lvp(iLevel) - T.lvp(iLevel-1);
        for block = 1:nblocks
            parentIdx = T.lvp(iLevel-1) + block;
            % If this block doesn't contain a jChild-th child, there is
            % nothing to do
            if length(T.nodes(parentIdx).chld) < max(jChild,2)
                continue;
            end
            jNodeIdx = T.nodes(parentIdx).chld(jChild);
            xj = T.nodes(jNodeIdx).xi;
            if ~redo_flag
                Y(xj,:) = R{iLevel}(xj,1:nSample(iLevel));
            else
                Y(xj,:) = R{iLevel}(xj,1:nSample(iLevel)/2);
            end
        end
        
        % Probe with Y and subtract off the low-rank approximations from
        % previous levels
        B  = zeros(size(Y));
        B2 = zeros(size(Y));
        
        n_probe = size(Y,2);
        
        for p = 1:ceil(n_probe/BLOCK_SIZE);
            crnt_idx = (p-1)*BLOCK_SIZE+1:min(p*BLOCK_SIZE,n_probe);
            B(:,crnt_idx)  = Afun(Y(:,crnt_idx));
            B2(:,crnt_idx) = apply_partial_peel(GHat,T,Y(:,crnt_idx),x,iLevel-1);
        
        end
        
        B = B-B2;

        if verb
            v_total = v_total + size(Y,2);
        end
        
        for block = 1:nblocks
            % Extract the effect of sources here on their siblings
            parentIdx = T.lvp(iLevel-1) + block;
            
            for iChild = 1:nSplit
                if iChild == jChild || length(T.nodes(parentIdx).chld) < max([iChild,jChild,2])
                    continue;
                end

                iNodeIdx = T.nodes(parentIdx).chld(iChild);
                jNodeIdx = T.nodes(parentIdx).chld(jChild);
                
                xi = T.nodes(iNodeIdx).xi;
                
                I = iNodeIdx - T.lvp(iLevel);
                J = jNodeIdx - T.lvp(iLevel);
                % prepend the result so that when its a redo we have the
                % new ones at the front
                if isempty(GR{I,J})
                    GR{I,J} = B(xi,:);
                else
                    GR{I,J} = [B(xi,1:nSample(iLevel)/2)  GR{I,J}];
                end
                
            end

        end
        
    end
    
    % Use the random samples to get an approximate SVD of each off-diagonal
    % submatrix
    
    % This flag will be used to break out of multiple loops if we have to
    % redo this level
    redo_flag = 0;
    
    for block = 1:nblocks
        parentIdx = T.lvp(iLevel-1) + block;
        if length(T.nodes(parentIdx).chld) < 2;
            %We are a leaf
            continue;
        end
        for iChild = 1:nSplit
            % if we don't have an iChild, skip
            if length(T.nodes(parentIdx).chld) < iChild
                continue;
            end
            iNodeIdx = T.nodes(parentIdx).chld(iChild);
            xi = T.nodes(iNodeIdx).xi;
            I = iNodeIdx - T.lvp(iLevel);
            
            for jChild = iChild+1:nSplit
                % if we don't have a jChild, skip
                if length(T.nodes(parentIdx).chld) < jChild
                    continue;
                end
               
                jNodeIdx = T.nodes(parentIdx).chld(jChild);
                xj = T.nodes(jNodeIdx).xi;
                J = jNodeIdx - T.lvp(iLevel);
                
                % We will use symmetry to get IJ and JI blocks at the same time                
                R1 = R{iLevel}(xj,:);
                R2 = R{iLevel}(xi,:);
                
                % SVD of A_{I,J}R1
                [U1,S1,~] = svd(GR{I,J},'econ');
                S1 = diag(S1);
              
                
                % SVD of A^T_{I,J}R2 = A_{J,I}R2
                [U2,S2,~] = svd(GR{J,I},'econ');
                 S2 = diag(S2);


                % Compute rank r
                r = max([find(S1 < S1(1)*tol,1), find(S2 < S2(1)*tol,1)]);
                %r = min(r, maxRank(iLevel));
                
                % check this
                if isempty(r)
                    % Can't do better if our blocks are small
                    maxsize = min(length(xi),length(xj));
                    if length(S1) >= maxsize && length(S2) >= maxsize
                        r = maxsize;
                    else
                        redo_flag = 1;
                        break;
                    end
                else
                    r = min([r, length(S1), length(S2)]);
                end
                
                r_avg        = r_avg + r;
                r_min        = min(r,r_min);
                r_max        = max(r,r_max);
                n_r_computed = n_r_computed + 1;
                
                U1 = U1(:,1:r);
                U2 = U2(:,1:r);
                
                GHat{iLevel}(I,J).U = U1;
                GHat{iLevel}(J,I).U = U2;
                
                % Don't double-store M.  Can always store 
                % things just at the sorted tuple (min, max)

                M = pinv(R2' * U1) * (R2'* GR{I,J}) * ...
                    pinv(U2' * R1);

                minIdx = min(I,J);
                maxIdx = max(I,J);
                
                if minIdx == I
                    GHat{iLevel}(minIdx,maxIdx).M = M;
                else
                    GHat{iLevel}(minIdx,maxIdx).M = M';
                end
            end
            if (redo_flag)
                break;
            end
        end
        if (redo_flag)
            break;
        end
    end
    if (redo_flag)
        if verb
            disp('Add sampling for current level...');
        end
        R{iLevel} = [randn(N,nSample(iLevel)) R{iLevel}];

        maxRank(iLevel) = 2*maxRank(iLevel);
        % This doubles the oversampling amount, but whatever
        nSample(iLevel) = 2*nSample(iLevel);

        % don't increase loop index, just go around again
        continue;
    end
    
        clear GR;
        % Can't clear an entry of an array so we explicitly empty it
        R{iLevel} = [];
    
    if verb
        temp = toc(lvlStart);
        t_total = t_total + temp;
        fprintf('%3d | %3.0f (%3.0f, %3.0f) | %10.2e (s)\n', ...
              iLevel, r_avg / n_r_computed, r_min, r_max, temp);
    end
    
     maxRanksObserved(iLevel) = r_max;
    % Increase loop index for next iteration
    iLevel = iLevel + 1;
end % end peeling

lvlStart = tic();

% Need to extract the diagonal i.e. any block with no children by probing with the identity
n_leaf  = 0;
max_size = 0;

% How many leaf nodes are there?  and what is the maximum size of a leaf
% node?
for i = 1:T.lvp(end)
    if isempty(T.nodes(i).chld)
        n_leaf = n_leaf + 1;
        max_size = max(max_size, length(T.nodes(i).xi));
    end
end

% first component will be matrix, second will be index of box in T.
G_D = cell(n_leaf,2);

Y = zeros(N, max_size);
currentidx = 1;
for i = 1:T.lvp(end)
    if ~isempty(T.nodes(i).chld)
        continue;
    end
    G_D{currentidx,2} = i;
    currentidx = currentidx + 1;
    
    xi = T.nodes(i).xi;
    Y(xi,1:length(xi)) = eye(length(xi));
end


B  = zeros(size(Y));
B2 = zeros(size(Y));

n_probe = size(Y,2);

for p = 1:ceil(n_probe/BLOCK_SIZE);
    crnt_idx = (p-1)*BLOCK_SIZE+1:min(p*BLOCK_SIZE,n_probe);
    B(:,crnt_idx)  = Afun(Y(:,crnt_idx));
    B2(:,crnt_idx) = apply_partial_peel(GHat,T,Y(:,crnt_idx),x,nLevel);

end

B = B-B2;

if verb
    v_total = v_total + size(Y,2);
end


for iBox = 1:n_leaf
    iNodeIdx = G_D{iBox,2};
    xi = T.nodes(iNodeIdx).xi;
    G_D{iBox,1} = B(xi,1:length(xi));
end


% Definitely return the trace
trace_est = trace_peel(G_D);


% Optionally return the observed ranks
nout = max(nargout,1) - 1;
for k = 1:nout
   varargout{k} = maxRanksObserved;
end


if verb
    temp = toc(lvlStart);
    t_total = t_total + temp;
    fprintf(['-'*ones(1,80) '\n'])
    fprintf(sprintf('Elapsed time is %f seconds.\n',t_total));
    fprintf(['-'*ones(1,80) '\n'])
    fprintf(sprintf('Applied matrix %d times.\n',v_total));
    fprintf(['-'*ones(1,80) '\n'])
end

end % end function

