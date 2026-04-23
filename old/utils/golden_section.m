% runs golden section on a function and returns all of the intermediate
% queries and evaluations; designed to heuristically use a coarse initial
% grid of evaluations
function [grid, evals] = golden_section(eval_func, start_grid, start_evals, N)

n = length(start_grid);
grid = zeros(N, 1); grid(1:n) = start_grid;
evals = zeros(N, 1); evals(1:n) = start_evals;

best = min(start_evals);
argmins = find(start_evals == best);
left = max(1, min(argmins)-1);
right = min(n, max(argmins)+1);
oleft = start_grid(left);
oright = start_grid(right);

tau = (sqrt(5)-1)/2;
if right-left == 2
    ocenter = start_grid(left+1);
    n = n+1;
    if oright-ocenter > ocenter-oleft
        o1 = ocenter; f1 = start_evals(left+1);
        o2 = oleft + tau*(oright-oleft); grid(n) = o2;
        f2 = eval_func(o2); evals(n) = f2;
    elseif oright-ocenter < ocenter-oleft
        o1 = oleft + (1-tau)*(oright-oleft); grid(n) = o1;
        f1 = eval_func(o1); evals(n) = f1;
        o2 = ocenter; f2 = start_evals(left+1);
    else
        o1 = oleft + (1-tau)*(oright-oleft); grid(n) = o1;
        f1 = eval_func(o1); evals(n) = f1;
        n = n+1;
        o2 = oleft + tau*(oright-oleft); grid(n) = o2;
        f2 = eval_func(o2); evals(n) = f2;
    end
elseif right-left == 3
    o1 = start_grid(left+1); f1 = start_evals(left+1);
    o2 = start_grid(right-1); f2 = start_evals(right-1);
else
    n = n+1;
    o1 = oleft + (1-tau)*(oright-oleft); grid(n) = o1;
    f1 = eval_func(o1); evals(n) = f1;
    n = n+1;
    o2 = oleft + tau*(oright-oleft); grid(n) = o2;
    f2 = eval_func(o2); evals(n) = f2;
end

for i = n+1:N
  if (f1 > f2) || (f1 == f2 && rem(i, 2))
    oleft = o1;
    o1 = o2;
    f1 = f2;
    o2 = oleft + tau*(oright-oleft); grid(i) = o2;
    f2 = eval_func(o2); evals(i) = f2;
  else
    oright = o2;
    o2 = o1;
    f2 = f1;
    o1 = oleft + (1-tau)*(oright-oleft); grid(i) = o1;
    f1 = eval_func(o1); evals(i) = f1;   
  end
end

end