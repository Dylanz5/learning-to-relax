% computes and plots the number of iterations required for SOR to converge,
% as well as the near-asymptotic bound for comparison, average across forty
% different randomly sampled linear systems, where the randomness
% determines a scalar offset of the diagonal

addpath ../solvers 
addpath ../utils

A = delsq(numgrid('S', 12));
n = length(A);
D = diag(diag(A));
L = tril(A,-1);
epsilon = 1E-8;
trials = 40;
omegas = omega_grid(A, 1., 1.99, .01);

% low-variance offset distribution
cs = -.15 + .6 * betarnd(2., 6., trials, 1);
taus = zeros(size(cs));
betas = zeros(size(cs));
actual = zeros(size(omegas));
dynamic_actual = 0.0;
predicted = zeros(size(omegas));
dynamic_predicted = 0.0;
for i = 1:length(cs)
    Ac = A + cs(i)*speye(n);
    b = truncated_normal(n);
    D = diag(diag(Ac));
    L = tril(Ac,-1);
    radius = zeros(size(omegas));
    errs = zeros(size(omegas));
    current_best = inf;
    for j = 1:length(omegas)
        k = sor(Ac, b, zeros(n, 1), omegas(j), epsilon);
        C = full(eye(n) - Ac * inv(D/omegas(j)+L));
        radius(j) = max(abs(eig(C)));
        errs(j) = norm(C^(k-1))^(1/(k-1)) - radius(j);
        actual(j) = actual(j) + k;
        if k < current_best
            current_best = k;
        end
    end
    dynamic_actual = dynamic_actual + current_best;
    betas(i) = max(abs(eig(eye(n)-inv(D)*Ac)));
    taus(i) = max(rdivide(errs, 1-radius));
    current_predicted = 1 + rdivide(log(epsilon), log(radius + taus(i)*(1-radius)));
    dynamic_predicted = dynamic_predicted + min(current_predicted);
    predicted = predicted + current_predicted;
end

ax = gca(figure(1));
semilogy(omegas, 35 * ones(size(omegas)), 'Color', 'white');
hold on;
semilogy(omegas, actual / length(cs), 'LineWidth', 2, 'Color', [0, 0.4470, 0.7410]);
semilogy(omegas, dynamic_actual * ones(size(omegas)) / trials, 'LineWidth', 2, 'LineStyle', '--', 'Color', [0, 0.4470, 0.7410]);
semilogy(omegas, 35 * ones(size(omegas)), 'Color', 'white');
semilogy(omegas, predicted / length(cs), 'LineWidth', 2, 'Color', [0.8500, 0.3250, 0.0980]);
semilogy(omegas, dynamic_predicted * ones(size(omegas)) / trials, 'LineWidth', 2, 'LineStyle', '--', 'Color', [0.8500, 0.3250, 0.0980]);
ax.XAxis.FontSize = 16;
ax.YAxis.FontSize = 16;
ylabel('iterations', 'FontSize', 20)
xlabel('\omega', 'FontSize', 24);
set(gcf, 'PaperPosition', [0, 0, 7, 5]);
legend('', 'actual cost', '(instance-optimal)', '', 'near-asymptotic bound', '(instance-optimal)', 'Location', 'north', 'FontSize', 18);
print('plots/low_variance.png', '-dpng', '-r256');
hold off;

% high-variance offset distribution
cs = -.15 + .6 * betarnd(.5, 1.5, trials, 1);
taus = zeros(size(cs));
betas = zeros(size(cs));
actual = zeros(size(omegas));
dynamic_actual = 0.0;
predicted = zeros(size(omegas));
dynamic_predicted = 0.0;
for i = 1:length(cs)
    Ac = A + cs(i)*speye(n);
    b = truncated_normal(n);
    D = diag(diag(Ac));
    L = tril(Ac,-1);
    radius = zeros(size(omegas));
    errs = zeros(size(omegas));
    current_best = inf;
    for j = 1:length(omegas)
        k = sor(Ac, b, zeros(n, 1), omegas(j), epsilon);
        C = full(eye(n) - Ac * inv(D/omegas(j)+L));
        radius(j) = max(abs(eig(C)));
        errs(j) = norm(C^(k-1))^(1/(k-1)) - radius(j);
        actual(j) = actual(j) + k;
        if k < current_best
            current_best = k;
        end
    end
    dynamic_actual = dynamic_actual + current_best;
    betas(i) = max(abs(eig(eye(n)-inv(D)*Ac)));
    taus(i) = max(rdivide(errs, 1-radius));
    current_predicted = 1 + rdivide(log(epsilon), log(radius + taus(i)*(1-radius)));
    dynamic_predicted = dynamic_predicted + min(current_predicted);
    predicted = predicted + current_predicted;
end

ax = gca(figure(2));
semilogy(omegas, 35 * ones(size(omegas)), 'Color', 'white');
hold on;
semilogy(omegas, actual / length(cs), 'LineWidth', 2, 'Color', [0, 0.4470, 0.7410]);
semilogy(omegas, dynamic_actual * ones(size(omegas)) / trials, 'LineWidth', 2, 'LineStyle', '--', 'Color', [0, 0.4470, 0.7410]);
semilogy(omegas, 35 * ones(size(omegas)), 'Color', 'white');
semilogy(omegas, predicted / length(cs), 'LineWidth', 2, 'Color', [0.8500, 0.3250, 0.0980]);
semilogy(omegas, dynamic_predicted * ones(size(omegas)) / trials, 'LineWidth', 2, 'LineStyle', '--', 'Color', [0.8500, 0.3250, 0.0980]);
ax.XAxis.FontSize = 16;
ax.YAxis.FontSize = 16;
ylabel('iterations', 'FontSize', 20)
xlabel('\omega', 'FontSize', 24);
set(gcf, 'PaperPosition', [0, 0, 7, 5]);
leg = legend('', 'actual cost', '(instance-optimal)', '', 'near-asymptotic bound', '(instance-optimal)', 'Location', 'north', 'FontSize', 18);
leg.Position(1) = leg.Position(1) + .1;
print('plots/high_variance.png', '-dpng', '-r256');
hold off;