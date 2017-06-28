% HYPOCT_TD_STRONG        Build top-down hyperoctree based on strong admissibility.
%
%    T = HYPOCT_TD_STRONG(X,OCC) builds a hyperoctree T over a set of points X such that
%    each nonempty hypercube node in T is recursively subdivided until it
%    contains at most OCC points. The tree T is structured as follows:
%
%        NLVL  - tree depth
%        LVP   - tree level pointer array
%        LRT   - size of tree root
%
%    It also contains the tree node data array NODES, with structure:
%
%        CTR   - node center
%        XI    - node point indices
%        PRNT  - node parent
%        CHLD  - node children
%        SIBL  - node siblings
%        NBOR  - node nieghbors
%        INTER - node interaction list
%
%    Some examples of how to access the tree data are given below:
%
%      - The nodes on level I are T.NODES(T.LVP(I)+1:T.LVP(I+1)).
%      - The size of each node on level I is T.LRT/2^(I-1).
%      - The points in node index I are X(:,T.NODES(I).XI).
%      - The parent of node index I is T.NODES(T.NODES(I).PRNT).
%      - The children of node index I are [T.NODES(T.NODES(I).CHLD)].
%      - The siblings of node index I are [T.NODES(T.NODES(I).SIBL)].
%      - The neighbors of node index I are [T.NODES(T.NODES(I).NBOR)].
%      - The nodes in the interaction list of node index I are [T.NODES(T.NODES(I).INTER)].
%
%    T = HYPOCT_TD_STRONG(X,OCC,LVLMAX) builds a hyperoctree to a maximum depth LVLMAX
%    (default: LVLMAX = INF).
%
%    T = HYPOCT_TD_STRONG(X,OCC,LVLMAX,EXT) sets the root node extent to
%    [EXT(I,1) EXT(I,2)] along dimension I. If EXT is empty (default), then the
%    root extent is calculated from the data.
%
%    References:
%
%      H. Samet. The quadtree and related hierarchical data structures. ACM
%        Comput. Surv. 16 (2): 187-260, 1984.

function T = hypoct_td_strong(x,occ,lvlmax,ext)

  % set default parameters
  if nargin < 3 || isempty(lvlmax)
    lvlmax = Inf;
  end
  if nargin < 4
    ext = [];
  end

  % check inputs
  assert(occ >= 0,'FLAM:hypoct_td:negativeOcc', ...
         'Leaf occupancy must be nonnegative.')
  assert(lvlmax >= 1,'FLAM:hypoct_td:invalidLvlmax', ...
         'Maximum tree depth must be at least 1.')

  % initialize
  [d,n] = size(x);
  if isempty(ext)
    ext = [min(x,[],2) max(x,[],2)];
  end
  l = max(ext(:,2) - ext(:,1));
  ctr = 0.5*(ext(:,1) + ext(:,2));
  s = struct('ctr',ctr','xi',1:n,'prnt',[],'chld',[],'sibl',[],'nbor',[],'inter',[]);
  T = struct('nlvl',1,'lvp',[0 1],'lrt',l,'nodes',s);
  nlvl = 1;
  nbox = 1;
  mlvl = 1;
  mbox = 1;

  % loop over all boxes in the tree
  while 1

    % terminate if at maximum depth
    if nlvl >= lvlmax
      break
    end

    % initialize level
    nbox_ = nbox;
    l = 0.5*l;

    % loop over all boxes at current level
    for prnt = T.lvp(nlvl)+1:T.lvp(nlvl+1)
      xi = T.nodes(prnt).xi;
      xn = length(xi);

      % subdivide box if it contains too many points
      if xn > occ
        ctr = T.nodes(prnt).ctr;
        idx = bsxfun(@gt,x(:,xi),ctr');
        idx = 2.^((1:d) - 1)*idx + 1;
        for i = unique(idx)
          nbox = nbox + 1;
          while mbox < nbox
            e = cell(mbox,1);
            s = struct('ctr',e,'xi',e,'prnt',e,'chld',e,'sibl',e,'nbor',e,'inter',e);
            T.nodes = [T.nodes; s];
            mbox = 2*mbox;
          end
          s = struct( 'ctr', ctr + l*(bitget(i-1,1:d) - 0.5), ...
                       'xi', xi(idx == i),                    ...
                     'prnt', prnt,                            ...
                     'chld', [],                              ...
                     'sibl', [],                              ...
                     'nbor', [],                              ...
                     'inter', []);
          T.nodes(nbox) = s;
          T.nodes(prnt).chld = [T.nodes(prnt).chld nbox];
        end
      end
    end

    % terminate if no new boxes added; update otherwise
    if nbox <= nbox_
      break
    else
      nlvl = nlvl + 1;
      T.nlvl = nlvl;
      while mlvl < nlvl
        T.lvp = [T.lvp zeros(1,mlvl)];
        mlvl = 2*mlvl;
      end
      T.lvp(nlvl+1) = nbox;
    end
  end

  % memory cleanup
  T.lvp = T.lvp(1:nlvl+1);
  T.nodes = T.nodes(1:nbox);
  
  % Put in sibling information
  for prnt = 1:nbox
      % connect all children
      chld = T.nodes(prnt).chld;
      for idx = chld
          T.nodes(idx).sibl = chld(chld~=idx);
      end
  end
  
  
   % initialize data for neighbor calculation
  ilvl = zeros(nbox,1);
  llvl = zeros(nlvl,1);
  l = T.lrt;
  for lvl = 1:nlvl
    ilvl(T.lvp(lvl)+1:T.lvp(lvl+1)) = lvl;
    llvl(lvl) = l;
    l = 0.5*l;
  end

  % find neighbors of each box
  for lvl = 2:nlvl
    l = llvl(lvl);
    for i = T.lvp(lvl)+1:T.lvp(lvl+1)
      ictr = T.nodes(i).ctr;
      prnt = T.nodes(i).prnt;

      % add all non-self children of parent
      j = T.nodes(prnt).chld;
      T.nodes(i).nbor = j(j ~= i);

      % add coarser parent-neighbors if adjacent
      for j = T.nodes(prnt).nbor
        if isempty(T.nodes(j).chld)
          jctr = T.nodes(j).ctr;
          dist = round((abs(ictr - jctr) - 0.5*(l + llvl(ilvl(j))))/l);
          if all(dist <= 0)
            T.nodes(i).nbor = [T.nodes(i).nbor j];
          end
        end
      end

      % add children of parent-neighbors if adjacent
      idx = [T.nodes(T.nodes(prnt).nbor).chld];
      c = reshape([T.nodes(idx).ctr],d,[])';
      dist = round(abs(bsxfun(@minus,T.nodes(i).ctr,c))/l);
      j = idx(max(dist,[],2) <= 1);
      if ~isempty(j)
        T.nodes(i).nbor = [T.nodes(i).nbor j];
      end

    end
  end
  
  for i = 1:length(T.nodes)
    T.nodes(i).nbor = [T.nodes(i).nbor i];
  end

  % find interaction list of each neighbor (level 4 is level 3 with 0
  % indexing)
  try
    assert(T.nlvl >= 4);
  catch
      error('Quadtree must have at least 4 levels for strong peeling.  Try adding more points.')
  end
  lvl = 4;
  % for each node on this first level, interaction list is range of this
  % level minus the things in the nbor list
  lvlrange = T.lvp(lvl)+1:T.lvp(lvl+1);
  for i = lvlrange
      T.nodes(i).inter = setdiff(lvlrange,T.nodes(i).nbor);
  end
  
  for lvl = 5:nlvl
       % for each node on this level, interaction list, parents nbors
       % children minus neighbors
       lvlrange = T.lvp(lvl)+1:T.lvp(lvl+1);
       for i = lvlrange
           prnt = T.nodes(i).prnt;
           idx = [T.nodes(T.nodes(prnt).nbor).chld];
           T.nodes(i).inter = setdiff(idx,T.nodes(i).nbor);
       end
  end
  

end