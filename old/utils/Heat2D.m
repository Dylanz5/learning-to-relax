% convenience class for solving the heat equation with a time-varying 
% diffusion coefficient using Crank-Nicholson on a uniform 2D grid
classdef Heat2D < handle

    properties
        coefficient
        forcing
        nx
        n
        dt
        t
        L
        I
        ijdx
        u
    end

    methods
        function obj = Heat2D(coefficient, forcing, initial, nx, dt)
            % coefficient: function handle of time
            % forcing: function handle of time and spatial coordinate
            % initial: function handle of spatial coordinate
            % nx: number of grid points in each dimension
            % dt: length of each time-step

            obj.coefficient = coefficient;
            obj.forcing = forcing;
            obj.nx = nx;
            obj.n = (nx-1)^2;
            dx = 1./nx;

            o = ones(nx-1, 1);
            X = spdiags([o, -4*o, o], [-1, 0, 1], nx-1, nx-1);
            [Xi, Xj, Xv] = find(X);
            Li = zeros(length(Xi) * (nx-1), 1);
            Lj = zeros(length(Li), 1);
            Lv = zeros(length(Li), 1);
            for i=1:nx-1
                Li((i-1)*length(Xi)+1:i*length(Xi)) = Xi + (i-1)*(nx-1);
                Lj((i-1)*length(Xj)+1:i*length(Xj)) = Xj + (i-1)*(nx-1);
                Lv((i-1)*length(Xv)+1:i*length(Xv)) = Xv;
            end
            o = ones(obj.n, 1);
            obj.L = (sparse(Li, Lj, Lv) + spdiags([o, o], [-nx+1, nx-1], obj.n, obj.n)) / dx^2;
            obj.I = speye(obj.n);

            obj.ijdx = zeros(obj.n, 2);
            k = 1;
            for i = 1:nx-1
                for j = 1:nx-1
                    obj.ijdx(k,:) = [i, j] * dx;
                    k = k+1;
                end
            end
            obj.u = initial(obj.ijdx);
            obj.t = 0.;
            obj.dt = dt;

        end

        % returns the current linear system that needs to be solved to
        % advance the simulation
        function [A, b] = crank_nicolson_system(obj)
            halfstep = obj.t + .5 * obj.dt;
            CL = .5 * obj.coefficient(halfstep) * obj.dt * obj.L;
            A = obj.I - CL;
            b = (obj.I + CL) * obj.u + obj.dt * obj.forcing(halfstep, obj.ijdx);
        end

        % updates the simulation using the provided vector
        function update(obj, u)
            obj.u = u;
            obj.t = obj.t + obj.dt;
        end

        % plots the temperature at each grid coordinate
        function show(obj)
            V = zeros(obj.nx+1, obj.nx+1);
            V(2:obj.nx, 2:obj.nx) = reshape(obj.u, obj.nx-1, obj.nx-1)';
            mesh((0:obj.nx)/obj.nx, (0:obj.nx) / obj.nx, V);
            axis([0, 1, 0, 1, 0, 1, 0, 1]);
        end
    
    end
end