
clear all; clc; pause(0.01), randn('seed',3212), rand('seed',3212), warning off

%--------------------------------------------------------------------------
% Adding folders
%--------------------------------------------------------------------------

addpath([pwd '/Data/']);

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

nsamples = 1e4;

% create store variables
beta_draw  = zeros(samples,p);
sigma_draw = zeros(samples,1);

XtX=X'*X;
Xty=X'*y;
V_inv       = Sigma_0\eye(size(Sigma_0));
P_post      = V_inv+XtX;
Sigma_post  = P_post\eye(size(P_post));

m_post      = Sigma_post*(V_inv*beta_0+Xty);
a_post      = a_0 +(n+p)*0.5;
b_post      = b_0 + 0.5 * sum((y - X*m_post).^2); 

% set counting variable for the number of iterations
no_samples=0;


while no_samples < samples 
     no_samples = no_samples + 1;
       
     % sigma draws 
     sig_inv = 1/b_post * randgamma(a_post,1); %random draw from a gamma (a_post, 1/b_post)
     sigma   = 1/sig_inv;
     
     % beta draws
     beta = mvrn(m_post,sigma*Sigma_post,1)';
     
     beta_draw(no_samples,:)=beta;
     sigma_draw(no_samples)=sigma;
end
     
figure(1)
histogram(beta_draw(:,1))
hold on
histogram(beta_draw(:,3))




