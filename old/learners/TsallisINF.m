% implements the Tsallis-INF bandit algorithm
% NOTE: uses a time-varying setting of eta = 2 / sqrt(t)
classdef TsallisINF < handle
    
    properties
        grid
        d
        t
        k
        index
        prob
        scale

        actions
        losses

    end
    
    methods
        function obj = TsallisINF(grid, T)
            % grid: action space
            % T: number of rounds (can be set to zero if not known)

            obj.grid = grid;
            obj.d = length(grid);
            obj.t = 1;
            obj.k = zeros(obj.d, 1);
            obj.scale = 1.;
            obj.actions = zeros(T, 1);
            obj.losses = zeros(T, 1);

        end
        
        % samples action to be taken
        function out = predict(obj)

            eta = 2. / sqrt(obj.t);
            x = -1.;
            for i = 1:20
                probs = 4 * (eta*(obj.k/obj.scale - x)).^(-2.);
                x = x - (sum(probs) - 1) / (eta * sum(probs.^1.5));
            end
            
            try
                obj.index = randsample(1:obj.d, 1, true, probs);
            catch
                obj.index = randsample(1:obj.d, 1, true);
                obj.prob = 1. / obj.d;
            end
            obj.prob = probs(obj.index);
            out = obj.grid(obj.index);

            obj.actions(obj.t) = obj.index;

        end

        % updates action distribution using the incurred cost
        function update(obj, loss)
            
            obj.losses(obj.t) = loss;
            obj.scale = mean(obj.losses(1:obj.t))-1.;
            obj.k(obj.index) = obj.k(obj.index) + (loss-1.) / obj.prob;
            obj.t(1) = obj.t(1) + 1;

        end
    end
end