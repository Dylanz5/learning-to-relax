% compares the performance of different learning algorithms---including 
% contextual bandit algorithms using diagonal offsets as context---on a 
% sequence of 5000 i.i.d. linear systems; all results are averages over 40 
% trials

addpath ../learners
addpath ../solvers 
addpath ../utils

A = delsq(numgrid('S', 12));
n = length(A);
epsilon = 1E-8;
T = 5000;
trials = 40;
tinf_costs = zeros(T, trials);
cheb_costs = zeros(T, trials);
tinfcb_costs = zeros(T, trials);
opt_costs = zeros(T, trials);
omega_costs = zeros(T, trials);

% high-variance offset distribution
parfor trial = 1:trials
    tinf = TsallisINF(linspace(1, 1.95, 20), T);
    cheb = ChebCB(linspace(1., 1.95, 20), T, 6, -.15, .65);
    tinfcb = TsallisINFCB(linspace(1, 1.95, 20), 5, -.15, .65);
    for t = 1:T
        c = -.15 + .6 * betarnd(.5, 1.5);
        At = A + c * speye(n);
        bt = truncated_normal(n);
        tinf_costs(t, trial) = sor(At, bt, zeros(n, 1), tinf.predict(), epsilon);
        tinf.update(tinf_costs(t, trial));
        cheb_costs(t, trial) = sor(At, bt, zeros(n, 1), cheb.predict(c), epsilon);
        cheb.update(cheb_costs(t, trial));
        tinfcb_costs(t, trial) = sor(At, bt, zeros(n, 1), tinfcb.predict(c), epsilon);
        tinfcb.update(tinfcb_costs(t, trial));
        opt_costs(t, trial) = sor(At, bt, zeros(n, 1), omega_opt(At), epsilon);
        omega_costs(t, trial) = sor(At, bt, zeros(n, 1), 1.8, epsilon);
    end
    fprintf('trial %2d finished\n', trial);
end

ax = gca(figure(1));
plot(mean(cumsum(tinf_costs), 2), T-(1:T), 'LineWidth', 2);
hold on;
plot(mean(cumsum(omega_costs), 2), T-(1:T), 'LineWidth', 2);
plot(mean(cumsum(tinfcb_costs), 2), T-(1:T), 'LineWidth', 2);
plot(mean(cumsum(cheb_costs), 2), T-(1:T), 'LineWidth', 2);
plot(mean(cumsum(opt_costs), 2), T-(1:T), 'LineWidth', 2);
legend('Tsallis-INF', '\omega=1.8 (\approx\omega^\ast)', 'Tsallis-INF-CB', 'ChebCB', 'Instance-Optimal', 'FontSize', 20);
ax.XAxis.FontSize = 16;
ax.YAxis.FontSize = 16;
xlabel('total iterations', 'FontSize', 20);
ylabel('instances remaining', 'FontSize', 20);
set(gcf, 'PaperPosition', [0, 0, 7, 5]);
print('plots/contextual_high_variance.png', '-dpng', '-r256');
hold off;

% low-variance offset distribution
parfor trial = 1:trials
    tinf = TsallisINF(linspace(1, 1.95, 20), T);
    cheb = ChebCB(linspace(1., 1.95, 20), T, 6, -.15, .65);
    tinfcb = TsallisINFCB(linspace(1, 1.95, 20), 5, -.15, .65);
    for t = 1:T
        c = -.15 + .6 * betarnd(2., 6.);
        At = A + c * speye(n);
        bt = truncated_normal(n);
        tinf_costs(t, trial) = sor(At, bt, zeros(n, 1), tinf.predict(), epsilon);
        tinf.update(tinf_costs(t, trial));
        cheb_costs(t, trial) = sor(At, bt, zeros(n, 1), cheb.predict(c), epsilon);
        cheb.update(cheb_costs(t, trial));
        tinfcb_costs(t, trial) = sor(At, bt, zeros(n, 1), tinfcb.predict(c), epsilon);
        tinfcb.update(tinfcb_costs(t, trial));
        opt_costs(t, trial) = sor(At, bt, zeros(n, 1), omega_opt(At), epsilon);
        omega_costs(t, trial) = sor(At, bt, zeros(n, 1), 1.6, epsilon);
    end
    fprintf('trial %2d finished\n', trial);
end

ax = gca(figure(2));
plot(mean(cumsum(tinf_costs), 2), T-(1:T), 'LineWidth', 2);
hold on;
plot(mean(cumsum(omega_costs), 2), T-(1:T), 'LineWidth', 2);
plot(mean(cumsum(tinfcb_costs), 2), T-(1:T), 'LineWidth', 2);
plot(mean(cumsum(cheb_costs), 2), T-(1:T), 'LineWidth', 2);
plot(mean(cumsum(opt_costs), 2), T-(1:T), 'LineWidth', 2);
legend('Tsallis-INF', '\omega=1.6 (\approx\omega^\ast)', 'Tsallis-INF-CB', 'ChebCB', 'Instance-Optimal', 'FontSize', 20);
ax.XAxis.FontSize = 16;
ax.YAxis.FontSize = 16;
xlabel('total iterations', 'FontSize', 20);
ylabel('instances remaining', 'FontSize', 20);
set(gcf, 'PaperPosition', [0, 0, 7, 5]);
print('plots/contextual_low_variance.png', '-dpng', '-r256');
hold off;