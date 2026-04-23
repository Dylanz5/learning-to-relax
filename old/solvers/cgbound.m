% computes a bound on the number of iterations required for
% SSOR-preconditioned CG to solve a linear system to a specified tolerance;
% the bound is derived from the condition number analysis of Axelsson
% (1994, Theorem 7.17)
function bound = cgbound(A, omega, epsilon)

D = diag(diag(A));
L = tril(A,-1);
invA = inv(A);
kappa = eigs(A, 1) * eigs(invA, 1);

alpha = real(eigs(D*invA, 1));
beta = max(real(eig(full((L*inv(D)*L'-.25*D)*invA))));
tmo = 2. - omega;
tmooo = rdivide(tmo, omega);
ic = rdivide(tmo, 1. + .25*tmo.*tmooo*alpha + beta*omega);

bound = 1. + rdivide(log(sqrt(kappa)/epsilon + sqrt(kappa/epsilon^2-1)), ...
                     -log(1. - rdivide(2., 1. + sqrt(rdivide(1., ic)))));

end