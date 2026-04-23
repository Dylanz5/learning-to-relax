% samples a radially-truncated standard Gaussian using rejection sampling
function sample = truncated_normal(n)

while true
    sample = normrnd(0., 1., n, 1);
    if norm(sample) <= sqrt(n)
        return
    end
end

end