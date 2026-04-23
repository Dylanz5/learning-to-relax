% implements the ChebCB contextual bandit algorithm for continuous
% one-dimensional contexts
% NOTE: uses a time-varying setting of eta that increases linearly in t
classdef ChebCB < handle

    properties
        grid
        T
        d
        m
        t
        eta
        actions
        theta
        features
        losses
        a
        bma
        scale

        predicted
        contexts

        chebyshev_coefs
        solver_options

    end

    methods
        function obj = ChebCB(grid, T, m, a, b)
            % grid: action space
            % T: number of rounds (can be set to zero if not known)
            % m: largest power of the Chebyshev approximation
            % a: lower bound on the context space
            % b: upper bound on the context space

            obj.grid = grid;
            obj.T = T;
            obj.d = length(grid);
            obj.m = m;
            obj.t = 1;
            obj.actions = zeros(T, 1);
            obj.theta = zeros(obj.d, m);
            obj.features = zeros(T, m);
            obj.losses = zeros(T, 1);
            obj.eta = 0.;
            obj.a = a;
            obj.bma = b - a;
            obj.scale = 0.;

            obj.predicted = zeros(T, 1);
            obj.contexts = zeros(T, 1);

            if m > 1
                obj.chebyshev_coefs = zeros(m-2, m);
                syms x;
                for j = 2:m-1
                    obj.chebyshev_coefs(j-1, 1:j+1) = coeffs(chebyshevT(j, x), 'All');
                end
            end

            obj.solver_options = optimoptions('lsqlin', 'Display', 'off');

        end

        function out = chebT(obj, x)

            out = ones(1, obj.m);
            for j = 1:obj.m-1
                switch j
                    case 1
                        out(1, j+1) = x;
                    otherwise
                        out(1, j+1) = polyval(obj.chebyshev_coefs(j-1,1:j+1), x);
                end
            end

        end

        % plots the current regression approximation of the cost of an
        % action over the entire context interval
        function show(obj, action)
            n = 100;
            contexts = linspace(obj.a, obj.a+obj.bma, n);
            features = zeros(n, obj.m);
            for i = 1:n
               features(i, :) = obj.chebT(2./obj.bma*(contexts(i)-obj.a)-1.)';
            end
            plot(contexts, 1. + obj.scale * features * obj.theta(action, :)');
        end

        % predicts which action should be taken given a context as input
        function out = predict(obj, context)

            feature = obj.chebT(2./obj.bma*(context-obj.a)-1.); 
            obj.features(obj.t, :) = feature;
            yhat = obj.theta * feature';
            [ystar, istar] = min(yhat);
            other = (1:obj.d) ~= istar;
            probs = zeros(obj.d, 1);
            probs(other) = 1 ./ (obj.d + obj.eta * (yhat(other)-ystar));
            probs(istar) = 1. - sum(probs(other));

            i = randsample(1:obj.d, 1, true, probs);
            obj.actions(obj.t) = i;
            out = obj.grid(i);

            obj.predicted(obj.t) = obj.theta(i,:) * feature';
            obj.contexts(obj.t) = context;

        end

        % updates the algorithm using the incurred cost
        function update(obj, loss)

            obj.losses(obj.t) = loss;

            K = max(obj.losses);
            L = (max(obj.losses) - min(obj.losses)) / obj.bma;
            N = 2. + 4. * obj.bma * L / K * (1. + log(obj.m));
            update = obj.scale ~= K * N;
            obj.scale = K * N;
            ub = [1./N, 2.*obj.bma*L/obj.scale ./ (1:obj.m-1)];
            lb = -ub;

            resnorm = 0.;
            for i = 1:obj.d
                if update || i == obj.actions(obj.t)
                    idx = obj.actions == i;
                    if sum(idx) > 0
                        obj.theta(i, :) = lsqminnorm(obj.features(idx, :), (obj.losses(idx)-1.) / obj.scale);
                        if any(abs(obj.theta(i, :)) > ub)
                             [obj.theta(i, :), resnorm] = lsqlin(obj.features(idx, :), (obj.losses(idx)-1.) / obj.scale, [], [], [], [], lb, ub, obj.theta(i, :), obj.solver_options);
                        end
                    end
                end
            end

            alpha = (pi+2./pi*log(2.*obj.m+1))/(2.*obj.scale*(obj.m+1))*obj.bma*L;
            R = sum((obj.predicted(1:obj.t)-obj.losses(1:obj.t)/K/N).^2);
            obj.eta = 2. * obj.t * sqrt(obj.d*obj.t / (R-resnorm + 2.*alpha^2*obj.t));
            obj.t(1) = obj.t(1) + 1;

        end
    end
end