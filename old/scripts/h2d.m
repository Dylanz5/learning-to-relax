% evaluates the runtime and number of iterations of different linear system
% solvers (and learned settings for them) while running 5000 steps of a
% two-dimensional heat equation simulation with time-varying diffusion

addpath ../learners
addpath ../solvers 
addpath ../utils

T = 5000; % number of simulation steps
stoptime = 5.; 
dt = stoptime / T;
nxs = 25 * 2 .^ (0:4); % discretizations (change 4 to 2 for shorter eval)
nnxs = length(nxs);
epsilon = 1E-8; % solver tolerance
omegas = [1., 1.3, 1.5, 1.75, 1.95]; % fixed omegas to evaluate
trials = 3; % number of evaluations for learning algorithms

% sets diffusion coefficient
coefficient = @(t) max(.1*sin(2.*pi*t), -10.*sin(2.*pi*t));
coeffs = coefficient(linspace(0., stoptime, T));
cmin = min(coeffs);
cmax = max(coeffs);
% sets forcing to be a smooth bump function moving in a circle
forcing = @(t, x) 32.*bump(x, [.5+cos(16.*pi*t)/4., .5+sin(16.*pi*t)/4.], .125);
% sets initial conditions to be a smooth bump function in the middle
initial = @(x) bump(x, [.5, .5], .25);

results = dictionary;
results{'CG'} = dictionary;
for omega = omegas
    results{string(omega)} = dictionary;
end
results{'Tsallis-INF'} = dictionary;
results{'Tsallis-INF-CB'} = dictionary;
results{'ChebCB'} = dictionary;
vals = values(results);
for i = 1:length(vals)
    vals{i}{'wallclock'} = zeros(nnxs, 1);
    vals{i}{'iterations'} = zeros(nnxs, 1);
end
results{'A\b'} = dictionary;
results{'A\b'}{'wallclock'} = zeros(nnxs, 1);
results{'Instance-Optimal'} = dictionary;
results{'Instance-Optimal'}{'iterations'} = zeros(nnxs, 1);
results{'Instance-Optimal'}{'actions'} = zeros(nnxs, T);
results{'Fixed Optimal'} = dictionary;
results{'Fixed Optimal'}{'iterations'} = zeros(nnxs, 1);
results{'Fixed Optimal'}{'actions'} = zeros(nnxs, 1);
results{'Tsallis-INF'}{'actions'} = zeros(nnxs, trials, T);
results{'ChebCB'}{'actions'} = zeros(nnxs, trials, T);

Ng = 12;
omega_grid = zeros(nnxs, T, Ng);
niter_grid = zeros(nnxs, T, Ng);
Nc = 96; g = linspace(1., 1.95, Nc);
contours = zeros(nnxs, T, Nc);
printerval = 500;

rng(0);
for nxidx = 1:nnxs
    nx = nxs(nxidx);

    % evaluates runtime and output of MATLAB's exact solver as a baseline
    name = 'A\b';
    pde = Heat2D(coefficient, forcing, initial, nx, dt);
    tic;
    printstep = printerval;
    for i = 1:T
        [A, b] = pde.crank_nicolson_system();
        pde.update(A \ b);
        if i >= printstep
            fprintf(strcat('[n=%d] ', strrep(name, '\', '\\'), ': step %d / %d\n'), pde.n, i, T);
            printstep = printstep + printerval;
        end
    end
    results{name}{'wallclock'}(nxidx) = toc;
    delete(pde);

    % evaluates unpreconditioned CG
    name = 'CG';
    niters = 0; 
    pde = Heat2D(coefficient, forcing, initial, nx, dt);
    tic;
    printstep = printerval;
    for i = 1:T
        [A, b] = pde.crank_nicolson_system();
        [u, ~, ~, niter] = pcg(A, b, epsilon, 10000, [], [], pde.u);
        niters = niters + niter;
        pde.update(u);
        if i >= printstep
            fprintf(strcat('[n=%d] ', strrep(name, '\', '\\'), ': step %d / %d\n'), pde.n, i, T);
            printstep = printstep + printerval;
        end
    end
    results{name}{'wallclock'}(nxidx) = toc;
    results{name}{'iterations'}(nxidx) = niters;
    delete(pde);

    % evaluates SSOR-preconditioned CG at different settings of omega
    for omegaidx = 1:length(omegas)
        omega = omegas(omegaidx);
        if nx > 400 && omega ~= 1. && omega ~= 1.5
            continue
        end
        name = string(omega);
        niters = 0; 
        iterations = zeros(T, 1);
        pde = Heat2D(coefficient, forcing, initial, nx, dt);
        tic;
        printstep = printerval;
        for i = 1:T
            [A, b] = pde.crank_nicolson_system();
            [iterations(i), u] = ssor_pcg(A, b, pde.u, omega, epsilon);
            niters = niters + iterations(i);
            pde.update(u);
            if i >= printstep
                fprintf(strcat('[n=%d] ', strrep(name, '\', '\\'), ': step %d / %d\n'), pde.n, i, T);
                printstep = printstep + printerval;
            end
        end
        results{name}{'wallclock'}(nxidx) = toc;
        results{name}{'iterations'}(nxidx) = niters;
        niter_grid(nxidx, :, omegaidx) = iterations;
        delete(pde);
    end

    if nx <= 400
    % finds the instance-optimal settings of omega and computes rough
    % approximations to the iteration count profile
    name = 'golden section';
    pde = Heat2D(coefficient, forcing, initial, nx, dt);
    printstep = printerval;
    for i = 1:T
        [A, b] = pde.crank_nicolson_system();
        eval_omega = @(omega) ssor_pcg(A, b, pde.u, omega, epsilon);
        [og, ng] = golden_section(eval_omega, omegas, niter_grid(nxidx, i, 1:length(omegas)), Ng);
        omega_grid(nxidx, i, :) = og; niter_grid(nxidx, i, :) = ng;
        contours(nxidx, i, :) = round(interp1(og, ng, g));
        pde.update(A \ b);
        if i >= printstep
            fprintf(strcat('[n=%d] ', strrep(name, '\', '\\'), ': step %d / %d\n'), pde.n, i, T);
            printstep = printstep + printerval;
        end
    end
    delete(pde);

    % evaluates always using the best fixed setting of omega
    name = 'Fixed Optimal';
    [~, omegaidx] = min(mean(contours(nxidx, :, :)));
    omega = g(omegaidx);
    results{name}{'actions'}(nxidx) = omega;
    if min(abs(omegas-omega)) == 0.
        results{name}{'iterations'}(nxidx) = results{string(omega)}{'iterations'}(nxidx);
    else
        niters = 0; 
        pde = Heat2D(coefficient, forcing, initial, nx, dt);
        printstep = printerval;
        for i = 1:T
            [A, b] = pde.crank_nicolson_system();
            [niter, u] = ssor_pcg(A, b, pde.u, omega, epsilon);
            niters = niters + niter;
            pde.update(u);
            if i >= printstep
                fprintf(strcat('[n=%d] ', strrep(name, '\', '\\'), ': step %d / %d\n'), pde.n, i, T);
                printstep = printstep + printerval;
            end
        end
        results{name}{'iterations'}(nxidx) = niters;
        delete(pde);
    end

    % evaluates always using the instance-optimal omega
    name = 'Instance-Optimal';
    mins = min(niter_grid(nxidx, :, :), [], 3);
    niters = 0; 
    pde = Heat2D(coefficient, forcing, initial, nx, dt);
    printstep = printerval;
    for i = 1:T
        [A, b] = pde.crank_nicolson_system();
        argmins = find(niter_grid(nxidx, i, :) == mins(i));
        omega = .5 * (omega_grid(nxidx, i, min(argmins)) + omega_grid(nxidx, i, max(argmins)));
        results{name}{'actions'}(nxidx, i) = omega;
        [niter, u] = ssor_pcg(A, b, pde.u, omega, epsilon);
        niters = niters + niter;
        pde.update(u);
        if i >= printstep
            fprintf(strcat('[n=%d] ', strrep(name, '\', '\\'), ': step %d / %d\n'), pde.n, i, T);
            printstep = printstep + printerval;
        end
    end
    results{name}{'iterations'}(nxidx) = niters;
    delete(pde);
    end

    % evaluates using Tsallis-INF to set omega
    name = 'Tsallis-INF';
    for trial = 1:trials
        tinf = TsallisINF(linspace(1., 1.95, 20), T);
        niters = 0; 
        pde = Heat2D(coefficient, forcing, initial, nx, dt);
        tic;
        printstep = printerval;
        for i = 1:T
            [A, b] = pde.crank_nicolson_system();
            [niter, u] = ssor_pcg(A, b, pde.u, tinf.predict(), epsilon);
            tinf.update(niter);
            niters = niters + niter;
            pde.update(u);
            if i >= printstep
                fprintf(strcat('[n=%d] ', strrep(name, '\', '\\'), ': trial %d / %d, step %d / %d\n'), pde.n, trial, trials, i, T);
                printstep = printstep + printerval;
            end
        end
        results{name}{'wallclock'}(nxidx, trial) = toc;
        results{name}{'iterations'}(nxidx, trial) = niters;
        for i = 1:T
            results{name}{'actions'}(nxidx, trial, i) = tinf.grid(tinf.actions(i));
        end
        delete(pde);
    end

    % evaluates Tsallis-INF-CB to set omega
    name = 'Tsallis-INF-CB';
    for trial = 1:trials
        tinfcb = TsallisINFCB(linspace(1., 1.95, 20), 6, cmin, cmax);
        niters = 0; 
        pde = Heat2D(coefficient, forcing, initial, nx, dt);
        tic;
        printstep = printerval;
        for i = 1:T
            [A, b] = pde.crank_nicolson_system();
            [niter, u] = ssor_pcg(A, b, pde.u, tinfcb.predict(pde.coefficient(pde.t)), epsilon);
            tinfcb.update(niter);
            niters = niters + niter;
            pde.update(u);
            if i >= printstep
                fprintf(strcat('[n=%d] ', strrep(name, '\', '\\'), ': trial %d / %d, step %d / %d\n'), pde.n, trial, trials, i, T);
                printstep = printstep + printerval;
            end
        end
        results{name}{'wallclock'}(nxidx, trial) = toc;
        results{name}{'iterations'}(nxidx, trial) = niters;
        delete(pde);
    end

    % evaluates using ChebCB to set omega
    name = 'ChebCB';
    for trial = 1:trials
        cheb = ChebCB(linspace(1., 1.95, 20), T, 6, cmin, cmax);
        niters = 0; 
        pde = Heat2D(coefficient, forcing, initial, nx, dt);
        tic;
        printstep = printerval;
        for i = 1:T
            [A, b] = pde.crank_nicolson_system();
            [niter, u] = ssor_pcg(A, b, pde.u, cheb.predict(pde.coefficient(pde.t)), epsilon);
            cheb.update(niter);
            niters = niters + niter;
            pde.update(u);
            if i >= printstep
                fprintf(strcat('[n=%d] ', strrep(name, '\', '\\'), ': trial %d / %d, step %d / %d\n'), pde.n, trial, trials, i, T);
                printstep = printstep + printerval;
            end
        end
        results{name}{'wallclock'}(nxidx, trial) = toc;
        results{name}{'iterations'}(nxidx, trial) = niters;
        for i = 1:T
            results{name}{'actions'}(nxidx, trial, i) = cheb.grid(cheb.actions(i));
        end
        delete(pde);
    end

end

% plots contour plot of iteration costs for nx=100, as well as the actions
% taken by Tsallis-INF, ChebCB, the fixed optimum, and instance optima
ax = gca(figure(1));
j = 3;
c = contour(1:T, linspace(1., 1.95, 96), squeeze(contours(3, :, :))');
i = 1;
levels{i} = ''; i=i+1;
hold on;
w = 25;
plot(1:T, results{'Fixed Optimal'}{'actions'}(j) * ones(T, 1), 'Color', 'black', 'LineStyle', ':', 'LineWidth', 3);
levels{i} = 'Best Fixed \omega'; i=i+1;
plot(1:T, movmean(squeeze(results{'Instance-Optimal'}{'actions'}(j, :, :)), w), 'Color', 'black', 'LineWidth', 3);
levels{i} = 'Instance-Optimal'; i=i+1;

tinf_mavg = movmean(squeeze(results{'Tsallis-INF'}{'actions'}(j, 3, :)), w);
tinf_mstd = movstd(squeeze(results{'Tsallis-INF'}{'actions'}(j, 3, :)), w);
plot(1:T, tinf_mavg, 'Color', "#77AC30", 'LineWidth', 3);
levels{i} = 'Tsallis-INF (±\sigma)'; i=i+1;
fill([1:T, fliplr(1:T)], [tinf_mavg'-tinf_mstd', fliplr(tinf_mavg'+tinf_mstd')], [0.4660 0.6740 0.1880], 'FaceAlpha', .5, 'LineStyle', 'none');
levels{i} = ''; i=i+1;

cheb_mavg = movmean(squeeze(results{'ChebCB'}{'actions'}(j, 3, :)), w);
cheb_mstd = movstd(squeeze(results{'ChebCB'}{'actions'}(j, 3, :)), w);
plot(1:T, cheb_mavg, 'Color', "#A2142F", 'LineWidth', 3);
levels{i} = 'ChebCB (±\sigma)'; i=i+1;
fill([1:T, fliplr(1:T)], [cheb_mavg'-cheb_mstd', fliplr(cheb_mavg'+cheb_mstd')], [0.6350 0.0780 0.1840], 'FaceAlpha', .5, 'LineStyle', 'none');
levels{i} = ''; i=i+1;

legend(levels, 'FontSize', 14, 'Location', 'East');
cb = contourcbar;
cb.Label.VerticalAlignment = "middle";
cb.Label.Position = [-.5, cb.Limits(2)+2];
cb.Label.Rotation = 0.;
cb.Label.String = '#iterations';
cb.Label.FontSize = 16;
ax.XAxis.FontSize = 16;
ax.YAxis.FontSize = 16;
xlabel('simulation step', 'FontSize', 16);
ylabel('\omega', 'FontSize', 20);
ylim([1., 1.95]);
hold off;
print('plots/heat2d_contour.png', '-dpng', '-r256');

% plots the diffusion coefficient as a function of time
ax = gca(figure(2));
timesteps = linspace(0., stoptime, T+1);
plot(timesteps, coefficient(timesteps), 'LineWidth', 3);
legend('diffusion coefficient', 'FontSize', 20, 'Location', 'East');
ax.XAxis.FontSize = 16;
ax.YAxis.FontSize = 16;
xlabel('time', 'FontSize', 16);
print('plots/coefficient.png', '-dpng', '-r256');

% plots comparisons of iteration costs
ax = gca(figure(3));
labels = {'CG', '1', '1.5', 'Fixed Optimal', 'Instance-Optimal', 'Tsallis-INF', 'Tsallis-INF-CB', 'ChebCB'};
colors = {'#000000', "#4DBEEE", "#EDB120", "#A2142F", "#D95319", "#77AC30", "#0072BD", "#7E2F8E"};
styles = {'-+', '-v', '-^', ':', '--', '-o', '-s', '-d'};
for i = 1:5
    loglog((nxs(1:length(results{labels{i}}{'iterations'}))-1).^2, results{labels{i}}{'iterations'}, styles{i}, 'LineWidth', 3, 'Color', colors{i});
    hold on;
end
for i = 6:8
    loglog((nxs(1:length(results{labels{i}}{'iterations'}))-1).^2, mean(results{labels{i}}{'iterations'}, 2), styles{i}, 'LineWidth', 3, 'Color', colors{i});
    hold on;
end
labels{1} = 'vanilla CG';
labels{2} = '\omega=1';
labels{3} = '\omega=1.5';
labels{4} = 'Best Fixed \omega';
labels{5} = 'Instance-Optimal';
legend(labels, 'Location', 'NorthWest', 'FontSize', 14);
ax.XAxis.FontSize = 16;
ax.YAxis.FontSize = 16;
ax.XTick = [10^3 10^4 10^5];
xlabel('matrix dimension', 'FontSize', 16);
ylabel('#iterations', 'FontSize', 16);
xlim([min((nxs-1).^2), max((nxs-1).^2)]);
ylim([2E4, 2E6]);
hold off;
print('plots/iterations.png', '-dpng', '-r256');

% plots comparisons of wallclock costs, normalized by the wallclock of
% unpreconditioned CG
ax = gca(figure(4));
labels = {'CG', '1', '1.5', 'Tsallis-INF', 'ChebCB'};
colors = {"#4DBEEE", "#EDB120", "#A2142F", "#0072BD", "#7E2F8E"};
styles = {'-+', '-v', '-^', '-o', '-s'};
for i = 1:3
    semilogx((nxs-1).^2, results{labels{i}}{'wallclock'}./results{'CG'}{'wallclock'}, styles{i}, 'LineWidth', 3, 'Color', colors{i}, 'MarkerSize', 3+3*(i==1));
    hold on;
end
for i = 4:5
    semilogx((nxs-1).^2, mean(results{labels{i}}{'wallclock'}, 2)'./results{'CG'}{'wallclock'}, styles{i}, 'LineWidth', 3, 'Color', colors{i}, 'MarkerSize', 3);
end
labels{1} = 'vanilla CG';
labels{2} = '\omega=1';
labels{3} = '\omega=1.5';
labels{4} = 'Tsallis-INF';
labels{5} = 'ChebCB';
legend(labels, 'Location', 'NorthEast', 'FontSize', 22);
for i = 1:nnxs
    text(exp(log((nxs(i)-1)^2)-.06*(i-1)+(i<2)*.02-(i>4)*.25), 1.4-(i<3)*.57, string(round(results{'CG'}{'wallclock'}(i)/5000., 4)), 'FontSize', 10);
    text(exp(log((nxs(i)-1)^2)-.06*(i-1)+(i<2)*.02-(i>4)*.25), 1.22-(i<3)*.57, 'sec/step', 'FontSize', 10);
end
ax.XAxis.FontSize = 16;
ax.YAxis.FontSize = 16;
ax.XTick = [10^3 10^4 10^5];
xlabel('matrix dimension', 'FontSize', 16);
ylabel('wallclock / CG wallclock', 'FontSize', 16);
xlim([min((nxs-1).^2), max((nxs-1).^2)]);
hold off;
print('plots/wallclock.png', '-dpng', '-r256');

% plots comparisons of wallclock costs, normalized by the wallclock of
% unpreconditioned CG
ax = gca(figure(5));
labels = {'CG', '1', '1.5', 'Tsallis-INF', 'ChebCB'};
colors = {"#4DBEEE", "#EDB120", "#A2142F", "#0072BD", "#7E2F8E"};
styles = {'-+', '-v', '-^', '-o', '-s'};
for i = 1:3
    semilogx((nxs(3:nnxs)-1).^2, results{labels{i}}{'wallclock'}(3:nnxs)./results{'CG'}{'wallclock'}(3:nnxs), styles{i}, 'LineWidth', 3, 'Color', colors{i}, 'MarkerSize', 3+3*(i==1));
    hold on;
end
for i = 4:5
    errorbar((nxs(3:nnxs)-1).^2, mean(results{labels{i}}{'wallclock'}(3:nnxs,:), 2)'./results{'CG'}{'wallclock'}(3:nnxs), 2.*std(results{labels{i}}{'wallclock'}(3:nnxs,:)')./results{'CG'}{'wallclock'}(3:nnxs)/sqrt(trials), styles{i}, 'LineWidth', 3, 'Color', colors{i}, 'MarkerSize', 3);
end
labels{1} = 'vanilla CG';
labels{2} = '\omega=1';
labels{3} = '\omega=1.5';
labels{4} = 'Tsallis-INF';
labels{5} = 'ChebCB';
ax.XAxis.FontSize = 16;
ax.YAxis.FontSize = 16;
ax.XTick = [10^4 10^5 10^6];
xlabel('matrix dimension', 'FontSize', 16);
ylabel('wallclock / CG wallclock', 'FontSize', 16);
ylim([0, 1.75])
hold off;
print('plots/wallclock_bars.png', '-dpng', '-r256');