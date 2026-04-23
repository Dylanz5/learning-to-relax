% compares the number of iterations required to solve a linear system when
% the target vector is drawn from a truncated Gaussian vs. when it is
% generated using the smallest eigenvector of the iteration matrix for
% omega=1.4

addpath ../solvers
addpath ../utils

A = delsq(numgrid('S', 12));
n = length(A);
D = diag(diag(A));
L = tril(A,-1);
epsilon = 1E-8;

omegas = omega_grid(A, 1., 1.9, .001);
C = full(eye(n) - A * inv(D/1.4+L));
[vecs, ~] = eig(C);
degen = vecs(:, 99);
degen_cost = zeros(size(omegas));
gauss_cost = zeros(size(omegas));
gauss_std = zeros(size(omegas));
gauss_min = zeros(size(omegas));
gauss_max = zeros(size(omegas));

parfor i = 1:length(omegas)
    degen_cost(i) = sor(A, degen, zeros(n, 1), omegas(i), epsilon);
    costs = zeros(40, 1);
    for j = 1:length(costs)
        costs(j) = sor(A, truncated_normal(n), zeros(n, 1), omegas(i), epsilon);
    end
    gauss_cost(i) = mean(costs);
    gauss_max(i) = max(costs);
    gauss_min(i) = min(costs);
end

ax = gca(figure(1));
fill([omegas; flipud(omegas)], [gauss_min; flipud(gauss_max)], [1., .8, .8], 'LineStyle', 'none');
hold on;
plot(omegas, degen_cost, 'LineWidth', 2, 'Color', [0, 0.4470, 0.7410]);
plot(omegas, gauss_cost, 'LineWidth', 2, 'Color', [0.8500, 0.3250, 0.0980]	);
legend('', 'degenerate cost', 'mean cost', 'Location', 'north', 'FontSize', 18);
ax.XAxis.FontSize = 16;
ax.YAxis.FontSize = 16;
ylabel('iterations', 'FontSize', 20)
xlabel('\omega', 'FontSize', 24);
set(gcf, 'PaperPosition', [0, 0, 7, 5]);
axis([1., 1.9, -inf, inf]);
print('plots/degenerate.png', '-dpng', '-r256');
hold off;