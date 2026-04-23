% returns a grid of evenly spaced omegas plus the optimal omega for SOR
function grid = omega_grid(A, omega_min, omega_max, step)

opt = omega_opt(A);
before = floor((opt - omega_min) / step) + 1;
after = floor((omega_max - opt) / step) + 1;

grid = zeros(before+1+after, 1);
omega = omega_min;

for i = 1:before
    grid(i) = omega;
    omega = omega + step;
end

grid(before+1) = opt;

for i = before+2:before+after+1
    grid(i) = omega;
    omega = omega + step;
end

end