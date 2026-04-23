% computes the spectral radius of the Jacobi iteration matrix
function beta = rho_jacobi(A)

beta = max(abs(eig(full(eye(size(A, 1)) - inv(diag(diag(A))) * A))));

end