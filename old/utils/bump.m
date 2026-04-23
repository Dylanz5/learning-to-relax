% computes a bump function at the given point(s) with the specified center
% and radius
function out = bump(x, center, radius)

out = exp(1. - 1 ./ (1. - min(1, sum((x-center).^2 / radius^2, 2))));

end