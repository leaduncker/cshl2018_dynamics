function [A,B,history] = runADMM(maxiter,X,U,B0,lam,rho);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% get size of responses and input
N = size(X,1);

t_start = tic;
% constant defaults
QUIET    = 0;
ABSTOL   = 1e-4;
RELTOL   = 1e-4;

% Compute some statistics of responses and inputs
[XX0,XX1,UX1,UX0,UU0] = ComputeSufficientStats(X,U);

% initialise dual variables 
Lam = zeros(N,N);
Z   = zeros(N,N);

%B0 is the initial estimate of B

if ~QUIET
    fprintf('starting to run ADMM...\n')
    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
      'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
end

% first iteration uses B0
    % update A: Dynamics matrix
    A = (XX1 - B0*UX1 - Lam + rho*Z) / XX0;
    
    % update B: Input filters
    B = (UX0' - A*UX1') / UU0;
    
    % update Z: Auxiliary variable
    Z_old = Z;
    Z = shrinkage(A + 1/rho * Lam, lam/rho);
    
    % update Lam: Lagrange multiplier
    Lam = Lam + rho * (A - Z);

for k = 2:maxiter 
    t_iter = tic;
    % update A: Dynamics matrix
    A = (XX1 - B*UX1 - Lam + rho*Z) / XX0;
    
    % update B: Input filters
    B = (UX0' - A*UX1') / UU0;
    
    % update Z: Auxiliary variable
    Z_old = Z;
    Z = shrinkage(A + 1/rho * Lam, lam/rho);
    
    % update Lam: Lagrange multiplier
    Lam = Lam + rho * (A - Z);
    
    % compute objective function values 
    history.objval(k)  = costFunction(X,U,A,B,lam);

    history.r_norm(k)  = norm(A - Z, 'fro');
    history.s_norm(k)  = norm(-rho*(Z - Z_old),'fro');

    history.eps_pri(k) = sqrt(N^2)*ABSTOL + RELTOL*max(norm(A,'fro'), norm(Z,'fro'));
    history.eps_dual(k)= sqrt(N^2)*ABSTOL + RELTOL*norm(Lam,'fro');
    
    
    history.Z_l1norm(k) = sum(abs(Z(:)));
      
   if ~QUIET
        fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, ...
        history.r_norm(k), history.eps_pri(k), ...
        history.s_norm(k), history.eps_dual(k), history.objval(k));
    end
    
    if (history.r_norm(k) < history.eps_pri(k) && ...
       history.s_norm(k) < history.eps_dual(k))
         break;
    end
    
    
    % check convergence
         
    if ~QUIET
        fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, ...
            history.r_norm(k), history.eps_pri(k), ...
            history.s_norm(k), history.eps_dual(k), history.objval(k));
    end

    if (history.r_norm(k) < history.eps_pri(k) && ...
       history.s_norm(k) < history.eps_dual(k))
         break;
    end

    % heuistic to adjust dual gradient ascent rate rho:
    if history.r_norm(k) > 10*history.s_norm(k)
        rho=2*rho;
    elseif history.s_norm(k) > 10*history.r_norm(k)
        rho=rho/2;
    end
    history.iteration_time(k) = toc(t_iter);
        

end

if ~QUIET
    toc(t_start);
end


end