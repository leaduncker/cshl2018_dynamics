function [XX0,XX1,UX1,UX0,UU0] = ComputeSufficientStats(X,U);
%
% X is Neurons   x Time x Trials
% U is InputDims x Time x Trials
%

% Note: we have to be careful about wich indices to include

% compute lag zero covariance of responses
XX0 = vec2(X(:,2:end,:))*vec2(X(:,2:end,:))'; % < X_t X_t'> summed over trials and time from t=2:T

% compute lag one covariance of responses
XX1 = vec2(X(:,2:end,:))*vec2(X(:,1:end-1,:))';  % < X_t X_t-1'> summed over trials and time from t=2:T

% compute cross-covariance of inputs and responses

UX0 = vec2(U(:,2:end,:))*vec2(X(:,2:end,:))'; % zero lag
UX1 = vec2(U(:,2:end,:))*vec2(X(:,1:end-1,:))'; % lag one

% compute lag zero input covariance
UU0 = vec2(U(:,2:end,:))*vec2(U(:,2:end,:))';
