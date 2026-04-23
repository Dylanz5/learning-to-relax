% computes the asymptotically optimal omega for SOR
function opt = omega_opt(A)

beta = rho_jacobi(A);
opt = 1 + (beta / (1 + sqrt(1 - beta^2)))^2;

end