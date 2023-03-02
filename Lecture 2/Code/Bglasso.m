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
[n,p]       = size(x);      % size of the data set

% groups are assigned arbitrarily (just an example)
K           = 10;
temp        = repmat([1:K],K,1);groups = temp(:);

% ----------------Gibbs related preliminaries

nsave           = 5000;
nburn           = 1000;

ntot            = nsave + nburn;     % total number of draws
beta_draws      = zeros(nsave,p);    % betas
tau2_draws      = zeros(nsave,K);    % tau2 
lambda2_draws   = zeros(nsave,1);    % global shrinkage 
sigma2_draws    = zeros(nsave,1);    % residual variance

% ----------------Set priors

beta            = [x ; .1*eye(p)] \ [y; zeros(p,1)];   % set beta to a RR estimate
e               = y - x*beta;
sigma2          = e'*e/n;                  % initial sigma2 estimate
lambda1         = 1;

%==========================================================================
%====================| GIBBS ITERATIONS START HERE |=======================
%tic;
[gname,Ielm]    = unique(groups);    
K               = length(gname);        % groups labeling
tau2inv         = zeros(1,K);           % setup tau2inv
gsize           = zeros(1,K);           % group sizes
XtXg            = cell(1,K);            % precompute XtX
Igr             = cell(K,1);            % the actual grouping of covariates
    
    % for each group...
    for k = 1:K
        gID = groups(Ielm(k));        % group identifier
        Igr{k} = (groups==gID);
        
        tau2inv(k)=mean(1 ./ (beta(Igr{k}).^2));
        XtXg{k}=x(:,Igr{k})'*x(:,Igr{k});
        
        % group sizes
        gsize(k)=sum(Igr{k});
    end       
    
for irep = 1:ntot

    % 1. Update beta from Normal
    
    for k=1:K       % for each group
                Ainv_k             = (XtXg{k} + (tau2inv(k)*eye(gsize(k)))) \ eye(gsize(k));
                e_k                = y - (x*beta - x(:,Igr{k})*beta(Igr{k}))/2;
                mu                 = Ainv_k*x(:,Igr{k})'*e_k; % mean for group k
                Sigma              = sigma2 * Ainv_k; % covariance for group k             
                beta(Igr{k})       = mvrn(mu,Sigma,1);
    end
            
    % 2. Update of sigma2 from the inverse gamma
    shape = (n-1)/2 + p/2;                    % shape parameter
    e     = y - x*beta;

    scale = e'*e/2;
    for k= 1:K
        scale = scale + (beta(Igr{k})'*beta(Igr{k})*tau2inv(k))/2;
    end    
    sigma2 = 1/gamrnd(shape, 1/scale);

    % 3. Sample tau2 from the conditional    
    for k=1:K
        mu_hat=sqrt(lambda1^2 * sigma2 ./ beta(Igr{k})'*beta(Igr{k}));
        tau2inv(k)=igrng(mu_hat,lambda1^2);
    end
            
    shape   = (p + K)/2 + 1;
    scale   = sum(1./tau2inv)/2 + 0.1;
    lambda1 = sqrt(gamrnd(shape, 1/scale));    
            
    % Save draws
     if irep > nburn
         beta_draws(irep-nburn,:)    = beta';
         tau2_draws(irep-nburn,:)    = tau2inv;
         lambda2_draws(irep-nburn,:) = lambda1;
         sigma2_draws(irep-nburn,:)  = sigma2;
     end

end

beta_means = squeeze(mean(beta_draws,1))';
beta_vars  = squeeze(var(beta_draws,1))';
lambdas    = squeeze(mean(lambda2_draws,1))';
sigma      = squeeze(mean(sigma2_draws,1))';



