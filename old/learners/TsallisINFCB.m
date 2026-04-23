% implements a contextual bandit algorithm that discretizes the context 
% space and runs Tsallis-INF independently in each bin
classdef TsallisINFCB < handle

    properties
        grid
        disc
        action
    end

    methods

        function obj = TsallisINFCB(grid, m, a, b)
            % grid: action space
            % m: number of bins for discretizing the context space
            % a: lower bound on the context space
            % b: upper bound on the context space

            for i = 1:m
                handles(i) = TsallisINF(grid, 0);
            end
            obj.grid = handles;
            obj.disc = a + (b-a) * linspace(.5/m, 1.-.5/m, m);

        end
        
        % predicts which action should be taken given a context as input
        function out = predict(obj, context)
            [~, i] = min(abs(obj.disc - context));
            out = obj.grid(i).predict();
            obj.action = i;
        end

        % updates the algorithm using the incurred cost
        function update(obj, loss)
            obj.grid(obj.action).update(loss);
        end

    end
end