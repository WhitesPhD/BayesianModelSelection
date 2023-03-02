%% Implementation of the Bayesian student-t regression

clear all; clc; pause(0.01), randn('seed',3212), rand('seed',3212), warning off

% addpath([pwd '/Data/']);
% upload the data 
% data on the anomaly based portfolios are from 
% Dong, Xi, et al. "Anomalies and the expected market return." The Journal of Finance 77.1 (2022): 639-681.
% data on the equity premium (target variable) are from https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

Data_x      = readtable('dataLongShortMissing','ReadVariableNames',true);
Data_y      = readtable('dataFf3','ReadVariableNames',true);

% create target and predictors 
x           = table2array(Data_x(:,2:end));

% replace nan with the cross sectional average
for i = 1 : size(x,2)
    if sum(isnan(x(:,i))) > 0
    x(isnan(x(:,i)),i) = nanmean(x(isnan(x(:,i)),:),2);
    else
    continue   
    end
end

% standardise the target and the predictors
y           = zscore(Data_y.MKT);
x           = zscore(x);

lag         = 1;
y           = y(lag + 1 :end);
x           = x(1:end - lag,:);

[T, p]      = size(x);

nsave    = 5000;
nburn    = 1000;

% ----------------Gibbs related preliminaries

ntot            = nsave + nburn;     % total number of draws
beta_draws      = zeros(nsave,p);    % betas
tau2_draws      = zeros(nsave,p);    % tau2 
lambda2_draws   = zeros(nsave,1);    % global shrinkage 
sigma2_draws    = zeros(nsave,1);    % residual variance

% ----------------Set priors
% lambda2 ~ Gamma(r,d)

r       = 1;
delta   = 3;

% Get OLS quanities from the full model (only if degrees of freedom allow this)

if T > p
    
    beta_OLS = inv(x'*x)'*(x'*y);
    SSE_OLS = (y - x*beta_OLS)'*(y - x*beta_OLS);
    sigma2_OLS = SSE_OLS/(T-(p-1));

    beta    = beta_OLS;
    tau2    = 4*ones(p,1);
    D     = diag(tau2);
    lambda2 = (p*sqrt(sigma2_OLS)/(sum(abs(beta_OLS)))).^2;
    sigma2  = sigma2_OLS;

else
    
    beta     = 0*ones(p,1); 
    tau2     = 4*ones(p,1);
    D        = diag(tau2);
    lambda2  = 0.1; 
    sigma2   = 0.1; 
    
end

%==========================================================================
%====================| GIBBS ITERATIONS START HERE |=======================

for irep = 1:ntot

    % 1. Update beta from Normal
    A = inv(x'*x + inv(D));
    post_mean_beta = A*x'*y; %#ok<*MINV>
    post_var_beta = sigma2*A;
    beta       = mvrn(post_mean_beta,post_var_beta,1)';

    % 2. Update tau2_j from Inverse Gaussian
    for j = 1:p
        a1              = (lambda2*sigma2)./(beta(j,1).^2);
        a2              = lambda2;
        tau_inverse     = 1/a2 * randgamma(sqrt(a1),1); 
        tau2(j,1)       = 1/tau_inverse + 1e-15;        
    end
    
    % Now that we have the new estimate of tau2, update D (the prior
    % covariance matrix of beta)
    D = diag(tau2);
    
    % 3. Update lambda2 from Gamma
    b1 = p + r;
    b2 = 0.5*sum(tau2) + delta;
    lambda2 = b2*randgamma(b1,1);
    
    % 4. Update sigma2 from Inverse Gamma
    c1        = (T-1+p)/2;
    PSI       = (y-x*beta)'*(y-x*beta);
    c2        = 0.5*PSI + 0.5*(beta'/D)*beta;
    sig2_inv  = 1/c2 * randgamma(c1,1); 
    sigma2    = 1/sig2_inv + 1e-15;
            
    % Save draws
    if irep > nburn
        beta_draws(irep-nburn,:) = beta;
        tau2_draws(irep-nburn,:) = tau2;
        lambda2_draws(irep-nburn,:) = lambda2;
        sigma2_draws(irep-nburn,:) = sigma2;
    end
    
end

% collect the posterior mean estimates

beta_means = squeeze(mean(beta_draws,1))';
beta_vars  = squeeze(var(beta_draws,1))';
lambdas    = squeeze(mean(lambda2_draws,1))';
sigma      = squeeze(mean(sigma2_draws,1))';


figure(1)
histogram(lambda2_draws)




