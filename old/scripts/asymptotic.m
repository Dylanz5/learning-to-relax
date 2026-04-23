% plots several quantities associated with the near-asymptotic bound on the
% number of SOR iterations required to converge

addpath ../solvers 
addpath ../utils

A = delsq(numgrid('S', 12));
n = length(A);
D = diag(diag(A));
L = tril(A,-1);
b = truncated_normal(n);
epsilon = 1E-8;

% computes and plots different estimates of the number of SOR iterations, 
% as well as the asymptocity (difference betweeen spectral norm and
% spectral radius at convergence), all at different settings of omega

omegas = omega_grid(A, 1., 1.99, .01);
actual = zeros(size(omegas));
radius = zeros(size(omegas));
errs = zeros(size(omegas));
energy = zeros(size(omegas));

parfor i = 1:length(omegas)
    k = sor(A, b, zeros(n, 1), omegas(i), epsilon);
    C = full(eye(n) - A * inv(D/omegas(i)+L));
    radius(i) = max(abs(eig(C)));
    errs(i) = norm(C^(k-1))^(1/(k-1)) - radius(i);
    actual(i) = k;
    energy(i) = energy_norm(A, omegas(i));
end

tau = max(rdivide(errs, 1-radius));

ax = gca(figure(1));
semilogy(omegas, actual, 'LineWidth', 2);
hold on;
semilogy(omegas, rdivide(log(epsilon), log(radius)), 'LineWidth', 2, 'Color', [0.4660, 0.6740, 0.1880]);
semilogy(omegas, rdivide(log(epsilon / (2*sqrt(condest(A)))), log(energy)), 'LineWidth', 2, 'Color', [0.9290, 0.6940, 0.1250]);
semilogy(omegas, rdivide(log(epsilon), log(radius+tau*(1-radius))), 'LineWidth', 2, 'Color', [0.8500, 0.3250, 0.0980]);
legend('actual cost', 'asymptotic estimate', 'energy bound', 'near-asymptotic bound', 'Location', 'northwest', 'FontSize', 20);
ax.XAxis.FontSize = 16;
ax.YAxis.FontSize = 16;
ylabel('iterations', 'FontSize', 20)
xlabel('\omega', 'FontSize', 24);
set(gcf, 'PaperPosition', [0, 0, 7, 5]);
print('plots/bound_comparison.png', '-dpng', '-r256');
hold off;

ax = gca(figure(2));
axis square;
plot(omegas, errs, 'LineWidth', 2);
hold on;
plot(omegas, tau*(1-radius), 'LineWidth', 2);
xlabel('\omega');
leg = legend('||C_\omega^k||_2^{1/k} - \rho(C_\omega)', '\tau (1 - \rho(C_\omega))', 'Location', 'northwest', 'FontSize', 18);
ax.XAxis.FontSize = 16;
ax.YAxis.FontSize = 16;
xlabel('\omega', 'FontSize', 24);
set(gcf, 'PaperPosition', [0, 0, 7, 5]);
print('plots/asymptocity.png', '-dpng', '-r256');
hold off;

% computes values of tau and beta at different offsets of the original
% linear system matrix A by multiples of the identity

cs = linspace(-.15, .45, 97);
taus = zeros(size(cs));
betas = zeros(size(cs));
parfor i = 1:length(cs)
    Ac = A + cs(i)*speye(n);
    omegas = omega_grid(A, 1., 1.9, .01);
    D = diag(diag(Ac));
    L = tril(Ac,-1);
    radius = zeros(size(omegas));
    errs = zeros(size(omegas));
    for j = 1:length(omegas)
        k = sor(Ac, b, zeros(n, 1), omegas(j), epsilon);
        C = full(eye(n) - Ac * inv(D/omegas(j)+L));
        radius(j) = max(abs(eig(C)));
        errs(j) = norm(C^(k-1))^(1/(k-1)) - radius(j);
    end
    betas(i) = rho_jacobi(Ac);
    taus(i) = max(rdivide(errs, 1-radius));
end

ax = gca(figure(3));
plot(cs, betas, 'LineWidth', 2);
hold on;
plot(cs, 4/exp(2)*(1-1/exp(2)) * ones(size(cs)), 'LineWidth', 2, 'LineStyle', '--');
plot(cs, taus, 'LineWidth', 2);
plot(cs, ones(size(cs))/exp(2), 'LineWidth', 2, 'LineStyle', '--');
xlabel('c');
legend('\beta', '4(1-1/e^2)/e^2', '\tau', '1/e^2', 'Location', 'west', 'FontSize', 18);
ax.XAxis.FontSize = 16;
ax.YAxis.FontSize = 16;
xlabel('c', 'FontSize', 24);
axis([-inf inf 0. 1.]);
set(gcf, 'PaperPosition', [0, 0, 7, 5]);
print('plots/tau_beta.png', '-dpng', '-r256');
hold off;