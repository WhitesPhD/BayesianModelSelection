
%% The code is a simple version of 
% Korobilis (2017) “Quantile Regression Forecasts of Inflation under Model Uncertainty”
% there is no model uncertainty here

clear all; close all; clc;

% insert your data here
[data, txt] = xlsread('Put your data here');

%---------------------------| USER INPUT |---------------------------------
% Gibbs-related preliminaries
nsave = 2000;            % Number of draws to save
nburn = 2000;            % Number of draws to discard
ntot  = nsave + nburn;   % Number of total draws
iter  = 500;             % Print every "iter" iteration

%--------------------------------------------------------------------------

y     = data(:,1);
x     = data(:,2:end);    
[T,p] = size(x);

% ==============| Get OLS coefficients
beta_OLS = (y'/x')';
sigma_OLS = (y-x*beta_OLS)'*(y-x*beta_OLS)./(T-p-1);

% ==============| Define priors
% prior for beta ~ N(0,V)
V    = 9*eye(p);
Vinv = inv(V);

% prior for sigma2 ~ IG(a0,b0)
a0 = 0.1;
b0 = 0.1;

% ==============| Initialize vectors
quant = [5:5:95]./100;
n_q = length(quant);

beta = zeros(p,n_q);
z = ones(T,n_q);
sigma2 = zeros(1,n_q);
theta = zeros(1,n_q);  
tau_sq = zeros(1,n_q);

% ==============| Storage matrices
beta_draws = zeros(p,n_q,nsave);

for irep = 1:ntot

    for q = 1:n_q   % sample for each quantile
        tau_sq(:,q) = 2/(quant(q)*(1-quant(q)));
        theta(:,q) = (1-2*quant(q))/(quant(q)*(1-quant(q)));   
        
        % Sample regression variance sigma2
        a1 = a0 + 3*T/2;
        sse = (y-x*beta(:,q) - theta(:,q)*z(:,q)).^2;
        a2 = b0 + sum(sse./(2*z(:,q)*tau_sq(:,q))) + sum(z(:,q));       
        sigma2(1,q) = 1./gamrnd(a1,1./a2);
        
        % Sample regression coefficients beta      
        U = 1./(tau_sq(:,q).*z(:,q)) ;
        y_tilde = y - theta(:,q)*z(:,q);
        xsq = x.*repmat(U,1,size(x,2));
        V_beta = inv(xsq'*x + Vinv);
        miu_beta = V_beta*(xsq'*y_tilde);    
        beta(:,q) = miu_beta + chol(V_beta)'*randn(p,1);
    
        % Sample latent variables z_{t}
            k1 = sqrt(theta(:,q).^2 + 2*tau_sq(:,q))./abs(y-x*beta(:,q));
            k2 = (theta(:,q).^2 + 2*tau_sq(:,q))/tau_sq(:,q);
            z(:,q) = max(1./Draw_IG(k1,k2),1e-4);                       
    end
    
    if irep > nburn
       beta_draws(:,:,irep-nburn) = beta;
    end
end
