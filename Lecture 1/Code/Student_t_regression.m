
clear all; clc; pause(0.01), randn('seed',3212), rand('seed',3212), warning off

%--------------------------------------------------------------------------
% Adding folders
%--------------------------------------------------------------------------
addpath([pwd '/Data/']);
addpath([pwd '/functions/']);
addpath([pwd '/Bayesreg/']);

Data_anomalies        = readtable('dataLongShortMissing','ReadVariableNames',true);
Data_target           = readtable('dataFf3','ReadVariableNames',true);

X = table2array(Data_anomalies(:,2:end));

% replace nan with the cross sectional average
for i=1:size(X,2)
    if sum(isnan(X(:,i))) > 0
    X(isnan(X(:,i)),i) = nanmean(X(isnan(X(:,i)),:),2);
    else
    continue   
    end
end

y               = Data_target.MKT;
[X, muX, normX] = standardise(X);
X               = [ones(size(X,1),1) X];
[n, p]          = size(X);

% initial hyper-priors

Sigma_0  = eye(p)*1e4; % flat prior
beta_0   = zeros(p,1);

a_0      = 0.1;
b_0      = 10;

nsave    = 5000;
nburn    = 1000;

ntot            = nsave + nburn;     % total number of draws

% create store variables
beta_draws   = zeros(nsave,p);
sigma_draws  = zeros(nsave,1);
lambda_draws = zeros(nsave,n);

nu          = 3; 
sigma_inv   = 1/b_0*randgamma(a_0,1);
sigma       = 1/sigma_inv;
lambda       = nu/2*randgamma(nu/2,n);
nu1_post    = (nu+1)*0.5;
V_inv       = Sigma_0\eye(size(Sigma_0));
a_post      = a_0 +(n+p)*0.5;

% set counting variable for the number of iterations

for irep = 1:ntot
    
     XtX         = X'*diag(lambda)*X;
     Xty         = X'*diag(lambda)*y;

     % betas draws 
     P_post      = V_inv + sigma_inv*XtX;
     Sigma_post  = P_post\eye(size(P_post));
     m_post      = Sigma_post*(V_inv*beta_0+sigma_inv*Xty);
     beta        = mvrn(m_post,Sigma_post,1)'; %beta draw from corresponding full conditional

     % sigma draws 
     b_post    = (b_0+(y-X*beta)'*diag(lambda)*(y-X*beta))*0.5;
     sigma_inv = 1/b_post * randgamma(a_post,1); 
     sigma     = 1/sigma_inv;
     
     % lambda draws
     
     nu2_post  = 0.5*(nu+sigma_inv*(y-X*beta).^2);
     lambda     = nu2_post.*randgamma(nu1_post,n);
     
     
      if irep > nburn
      beta_draws(irep-nburn,:)   = beta;
      sigma_draws(irep-nburn)    = sigma;
      lambda_draws(irep-nburn,:) = lambda;
      end
          
end
     
beta_means = squeeze(mean(beta_draws,1))';
beta_vars  = squeeze(var(beta_draws,1))';
lambdas    = squeeze(mean(lambda_draws,1))';
sigma      = squeeze(mean(sigma_draws,1))';



