%% Implementation of the SSVS 
%Ročková, Veronika, and Edward I. George. "The spike-and-slab lasso." 
%Journal of the American Statistical Association 113.521 (2018): 431-444.

% Reference

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
gamma_draws     = zeros(nsave,p);    % tau2 
sigma2_draws    = zeros(nsave,1);    % residual variance
p_draws         = zeros(nsave,p);    % residual variance
tau0_draws      = zeros(nsave,p);    % residual variance
tau1_draws      = zeros(nsave,p);    % residual variance

% ----------------Set priors
% beta ~ N(0,DD), where D = diag(tau_i)

lambda_0 = 10;
lambda_1 = 0.1;

tau_0  = exprnd(lambda_0^2,p,1);
tau_1  = exprnd(lambda_1^2,p,1);

% gamma_j ~ Bernoulli(1,p_j)
p_j = 0.5*ones(p,1);
c1  = 0.1;
c2  = 0.1;

% Initialize parameters
gamma  = bernoullimrnd(p,p_j);       
D      = diag((1-gamma).*tau_0 + gamma.*tau_1);
sigma2 = Draw_iGamma(c1,c2);


% Initialize parameters
%==========================================================================
%====================| GIBBS ITERATIONS START HERE |=======================
%tic;
% fprintf('Now you are running SSVS')
% fprintf('\n')
% fprintf('Iteration 0000')
XtX = x'*x;
Xty = x'*y;

tic;
fprintf('Now you are running SSVS')
fprintf('\n')
fprintf('Iteration 0000')

for irep = 1:ntot
    
    if mod(irep,500) == 0
    fprintf('%c%c%c%c%c%c%c%c%c%c%c%c%c%c%s%4d',8,8,8,8,8,8,8,8,8,8,8,8,8,8,'Iteration ',irep)
    end
    
    
    % 1. Update beta from Normal
    D               = diag((1-gamma).*tau_0 + gamma.*tau_1);
    A               = inv(XtX + inv(D));
    post_mean_beta  = A*Xty; %#ok<*MINV>
    post_var_beta   = sigma2*A;
    beta            = Draw_Normal(post_mean_beta,post_var_beta);
        
    % 2. Update restriction indexes of alpha from Bernoulli
    u_i1 = normpdf(beta,0,sigma2*tau_0).*p_j;           
    u_i2 = normpdf(beta,0,sigma2*tau_1).*(1- p_j);
    gst = u_i2./(u_i1 + u_i2);
    gamma = bernoullimrnd(p,gst);       
    
    % 3. Update sigma2 from Inverse Gamma
    c1     = (T+p)/2;
    s      = (y-x*beta)'*(y-x*beta);
    c2     = 0.5*s + 0.5*(beta'/D)*beta;
    sigma2 = Draw_iGamma(c1,c2);
    
    % 4. Update pi from a Beta
    p_j   = betarnd(0.1 + sum(gamma), 0.1 + sum(1-gamma),p,1);    

    % 5. update tau2
    
    a = (lambda_0^2.*sigma2./beta.^2).^.5;
    b = (lambda_1^2.*sigma2./beta.^2).^.5;

    tau_0  = gamrnd(a,1/lambda_0^2); 
    tau_1  = gamrnd(b,1/lambda_1^2); 
    
    % Save draws
    if irep > nburn
        beta_draws(irep-nburn,:)   = beta;
        gamma_draws(irep-nburn,:)  = gamma;
        sigma2_draws(irep-nburn,:) = sigma2;
        p_draws(irep-nburn,:)      = p_j;
        tau0_draws(irep-nburn,:)   = tau_0;
        tau1_draws(irep-nburn,:)   = tau_1;
    end
    
end
fprintf('%c%c%c%c%c%c%c%c%c%c%c%c%c%c',8,8,8,8,8,8,8,8,8,8,8,8,8,8)
% fprintf('%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c',8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8)
toc; % Stop timer and print total time


beta_means = squeeze(mean(beta_draws,1))';
beta_vars  = squeeze(var(beta_draws,1))';
gammas     = squeeze(mean(gamma_draws,1))';
sigma      = squeeze(mean(sigma2_draws,1))';
p_j        = squeeze(mean(p_draws,1))';




