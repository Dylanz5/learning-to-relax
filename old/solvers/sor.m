% computes the number of iterations required for SOR to solver a linear
% system to a given tolerance, returning the approximate solution obtained
% as a second output; caps the number of iterations at 10000
function [k, out] = sor(A, b, x, omega, tol)

D = diag(diag(A));
M = D/omega + tril(A,-1);

r = b - A*x;
norm0 = norm(r);
for k = 1:10000
  x = x + M\r;
  r = b - A*x;
  if (norm(r)/norm0 < tol)
    out = x;
    return
  end
end
