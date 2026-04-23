% computes the number of iterations required for SSOR-preconditioned CG to 
% solver a linear system to a given tolerance, returning the approximate 
% solution obtained as a second output; caps the number of iterations at 
% 10000
function [k, out] = ssor_pcg(A, b, x, omega, tol)

D = diag(diag(A));
L = tril(A, -1);
X = D + omega * L;
[out, ~, ~, k] = pcg(A, b, tol, 10000, X'*inv(D), X/omega/(2.-omega), x);

end