% computes and plots the number of iterations required for
% SSOR-preconditioned CG to converge as well as the condition-number-based
% bound for comparison, all at different values of omega and for several
% different linear systems

addpath ../solvers
addpath ../utils

for region = ['L' 'S']
    for s = [12 32]
        for offset = [0. .5]

A = delsq(numgrid(region, s));
n = length(A);
A = A + offset*speye(n);
b = truncated_normal(n);
D = diag(diag(A));
L = tril(A,-1);

K = 100;
epsilon = 1E-8;
omegas = linspace(2.*sqrt(2.)-2., 1.9, K);
costs = zeros(size(omegas));
bounds = cgbound(A, omegas, epsilon) - 1.;

parfor i = 1:K
    x = zeros(n,1);
    omega = omegas(i);
    costs(i) = ssor_pcg(A, b, x, omegas(i), epsilon);
end

tau = max(rdivide(costs-1., bounds));

ax = gca(figure(1));
plot(omegas, costs, 'LineWidth', 2);
hold on;
plot(omegas, 1. + tau * bounds, 'LineWidth', 2, 'LineStyle', '--');
legend('actual cost', 'upper bound', 'Location', 'north', 'FontSize', 20);
title([region '-shaped domain: size=' int2str(n) ', offset=' num2str(offset)], 'Fontsize', 24)
ax.XAxis.FontSize = 16;
ax.YAxis.FontSize = 16;
ylabel('iterations', 'FontSize', 20)
xlabel('\omega', 'FontSize', 24);
set(gcf, 'PaperPosition', [0, 0, 7, 5]);
print(['plots/cgbound-' region '-' int2str(n) '-' num2str(offset) '.png'], '-dpng', '-r256');
hold off;

        end
    end
end