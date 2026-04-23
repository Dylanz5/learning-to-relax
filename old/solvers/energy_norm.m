% computes a bound on the energy norm of the SOR iteration matrix
% (Hackbusch 2016, Corollary 3.45)
function bound = energy_norm(A, omega)

Omega = (2 - omega) / (2*omega);
invD = inv(diag(diag(A)));
L = tril(A,-1);
gamma = 1. - max(abs(eig(invD*(L+L'))));
bound = sqrt(1. - 2*Omega*gamma / (Omega^2+gamma/omega + max(abs(eig(invD*L*invD*L')))-.25));

end