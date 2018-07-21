function obj = costFunction(X,U,A,B,lam)
%
% cost function for L1 penalised regression

% sum_t || X_t - A X_t-1 - B U_t||^2 + lam * ||A||_1

regressTerm = vec2(X(:,2:end,:)) - A*vec2(X(:,1:end-1,:)) - B*vec2(U(:,2:end,:)); % N x T*R
sparsityTerm = sum(abs(A(:)));

obj = sum(regressTerm(:).^2) + lam*sparsityTerm;

end