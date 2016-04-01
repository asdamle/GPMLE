% uses the factorization created by rskelf_mv_only, will fail otherwise!
function Y = rskelf_mv_h_fast(F,Y)

  % initialize
  n = F.lvp(end);
  N = size(Y,1);
  idx = cell(n,1);
  for i = 1:n
      sk = F.factors(i).sk;
      rd = F.factors(i).rd;
      idx{i} = [sk(:) ; rd(:)];
  end
 
  
  % do something with the fact that at a level sk and rd are disjoint
  for i = 1:n
      Y(idx{i},:) = F.factors(i).Aup*Y(idx{i},:);
  end
  
  % top level up and down factors are grouped in F.factors(n).Aup
  for i = (n-1):-1:1
      Y(idx{i},:) = F.factors(i).Adown*Y(idx{i},:);
  end
end