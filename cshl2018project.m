%% CSHL Computational Neuroscience: Vision 2018 project
%  Erika Dunn-Weiss

% Model: X_t = AX_t-1 + BU_t, for X_t timecourses, U_t stimuli 
clear all
N = 64;
nOris = 16; 
oris = [-pi/2:pi/nOris:pi/2];
Trials = round(100/size(oris,2))*nOris;
T = 15;
% Delta
delta = pi/16; % +/- 5 degrees about oblique reference (45)
% preferred orientations: assume pinwheel structure
theta_ix = repmat(1:nOris,N/nOris,1); %uniform distribution of preferred orientations 0:180
theta_ix = theta_ix(:);
theta = repmat(0:pi/8:7*pi/8,N/8,1);
theta = theta(:);
% feedforward input
Jf = 1.5;
sigma_f = pi/6;
generic_tc = Jf*exp(-oris.^2/(2*sigma_f^2));
for i = 1:N
    B(i,:) = generic_tc(circshift([1:length(oris)],theta_ix(i)-median([1:length(oris)]),2));
end
oris = oris(1:end-1);
B = B(:,1:16);
% recurrent input - assumptions
% 1. for now, just excitatory neurons, because I think that's all that I
% will label in 2P anyway
% 2. probability of synapse is determined by orientation preference alignment
% 3. the values of A represent the strength of the synapse, which for now
% is uniform 
sigma_fb = .3281/sqrt(2*log(2)); %conversion of HWHM to sigma from Roerig paper
delta_theta = abs(round(radian_distance(repmat(theta,1,N),repmat(theta,1,N)'),5));
dist = normpdf(delta_theta, 0, sigma_fb);  
normfunc = @(x) (1/(sqrt(2*pi*sigma_fb^2)))*exp(-(x.^2)/(2*sigma_fb^2));

A = zeros(N,N);
for i= 1:N^2
    A(i) = rand < integral(normfunc,delta_theta(i),inf);
end
A = A - diag(diag(A));
syn_strength = 0.15; 
A = syn_strength*A; 
%%
%%%%%%%%%  generate model parameters %%%%%%%%% 
uDim     = size(oris,2); %2 stimuli
xDim     = N; %64 neurons (for now)
Q0max    = 0.01;

Q  = diag(rand(xDim,1));
Q0 = dlyap(A,Q);
M  = diag(1./sqrt(diag(Q0)));
A  = M*A*pinv(M);
Q  = M*Q*M'; Q=(Q+Q)/2;

O  = orth(randn(xDim));
Q0 = O*diag(rand(xDim,1)*Q0max)*O'/3;
x0 = randn(xDim,1)/3;

params.model.A    = A;
params.model.Q    = Q;
params.model.Q0   = Q0;
params.model.x0   = x0;
params.model.Pi   = dlyap(params.model.A,params.model.Q);

params.model.B    = B; 
params.model.notes.useB = true;


%%%%%%%%%  sample fake data %%%%%%%%% 

u = repmat(eye(size(oris,2)),1,round(100/size(oris,2)));

%assignopts(who,varargin);

if numel(T)==1
   T = ones(Trials,1)*T;
end

Trials = numel(T);

CQ          = chol(params.model.Q);
CQ0         = chol(params.model.Q0);

for tr=1:Trials
  seq(tr).u = repmat(u(:,tr),1,unique(T));
  seq(tr).x = zeros(xDim,T(tr));
  seq(tr).x(:,1) = params.model.x0+CQ0'*randn(xDim,1);
  if params.model.notes.useB; seq(tr).x(:,1) = seq(tr).x(:,1)+params.model.B*seq(tr).u(:,1);end;
  for t=2:T(tr)
      seq(tr).x(:,t) = params.model.A*seq(tr).x(:,t-1)+CQ'*randn(xDim,1);
      if params.model.notes.useB; seq(tr).x(:,t) = seq(tr).x(:,t)+params.model.B*seq(tr).u(:,t);end;
  end
  seq(tr).T = T(tr);
  
  X(:,:,tr) = seq(tr).x;
  U(:,:,tr) = seq(tr).u;
end
%%
%%%%%%%%%  look at fake data %%%%%%%%% 
figure; subplot(2,2,1); imagesc(B); title('input weights'); xlabel('stim ori'); ylabel('neuron'); colorbar;
subplot(2,2,2); imagesc(A); title('recurrent weights'); xlabel('neuron'); ylabel('neuron'); colorbar;
for i = 1:Trials
    Trial.type(i) = find(seq(i).u(:,1));
    Trial.resp(:,i) = mean(seq(i).x,2);
end
for j = 1:length(unique(Trial.type))
    ix = find(Trial.type == j);
    Trial.mean_resp(:,j) = mean(Trial.resp(:,ix),2);
end
subplot(2,2,3); imagesc(seq(2).x); title('timecourse'); xlabel('time steps'); ylabel('neuron');
subplot(2,2,4); plot(Trial.mean_resp(1,circshift([1:16],8,2))); xlabel('ori'); ylabel ('response'); title('e.g. tuning curve');

%%
%%%%%%%%%  estimate A,B %%%%%%%%% 
for i = 1:Trials
    z = [X(:,:,i); U(:,:,i)];
    z1 = [X(:,1:end-1,i); U(:,1:end-1,i)];
    xz(:,:,i) = X(:,2:end,i)*z1';
    zz(:,:,i) = z*z';
end
XZ = sum(xz,3);
ZZ = sum(zz,3);
AB_est = XZ/ZZ;
A_est = AB_est(:,1:N);
B_est = AB_est(:,end-(length(oris)-1):end);
%%
% visualize estimation
figure; subplot(2,2,1); imagesc(B); colorbar; title('B')
subplot(2,2,2); imagesc(B_est); colorbar; title('B-est')
subplot(2,2,3); imagesc(A); colorbar; title('A')
subplot(2,2,4); imagesc(A_est); colorbar; title('A-est');

% assess success of A estimate
[u,s,v] = svd(A);
s = diag(s);
[ue,se,ve] = svd(A_est);
se = diag(se);
% projection onto the first few eigen vectors (dot product)
dot(u(:,1),ue(:,1))
dot(v(:,1),ve(:,1))
% comparison of top eigenvalues
%s(1:5) - se(1:5)

% assess success of B estimate
[u,s,v] = svd(B);
s = diag(s);
[ue,se,ve] = svd(B_est);
se = diag(se);
% projection onto the first few eigen vectors (dot product)
dot(u(:,1),ue(:,1))
dot(v(:,1),ve(:,1))
% comparison of top eigenvalues
%s(1:5) - se(1:5)
%%

% prediction - leave one out
for i = 1:Trials
    z = [X(:,1:end-1,i); U(:,1:end-1,i)];
    z1 = [X(:,1:end-1,i); U(:,1:end-1,i)];
    xz(:,:,i) = X(:,2:end,i)*z1';
    zz(:,:,i) = z*z';
end
XZ = sum(xz,3);
ZZ = sum(zz,3);
AB_est = XZ/ZZ;
A_est = AB_est(:,1:N);
B_est = AB_est(:,end-(length(oris)-1):end);
X_est = A_est*X(:,end-1,Trials)+CQ'*randn(xDim,1)+B_est*U(:,end-1,Trials);
[r,p] = corr(X_est,X(:,end,Trials))


%% sparsity 
% get estimates
[A_est, B_est, history] = runADMM(1000,X,U,B,100,1);

% visualize estimation
figure; subplot(2,2,1); imagesc(B); colorbar; title('B')
subplot(2,2,2); imagesc(B_est); colorbar; title('B-est')
subplot(2,2,3); imagesc(A); colorbar; title('A')
subplot(2,2,4); imagesc(A_est); colorbar; title('A-est')

% assess success of A estimate
[u,s,v] = svd(A);
s = diag(s);
[ue,se,ve] = svd(A_est);
se = diag(se);
% projection onto the first few eigen vectors (dot product)
dot(u(:,1),ue(:,1))
dot(v(:,1),ve(:,1))
% comparison of top eigenvalues
%s(1:5) - se(1:5)

% assess success of B estimate
[u,s,v] = svd(B);
s = diag(s);
[ue,se,ve] = svd(B_est);
se = diag(se);
% projection onto the first few eigen vectors (dot product)
dot(u(:,1),ue(:,1))
dot(v(:,1),ve(:,1))
% comparison of top eigenvalues
%s(1:5) - se(1:5)

%%
% prediction - leave one out
[A_est, B_est, history] = runADMM(1000,X(:,1:end-1,:), U(:,1:end-1,:),B,100,1);
X_est = A_est*X(:,end-1,Trials)+CQ'*randn(xDim,1)+B_est*U(:,end-1,Trials);
[r,p] = corr(X_est,X(:,end,Trials))

