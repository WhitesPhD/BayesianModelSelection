function [beta_draw]=ProbitMH(Y,X,v,tau,V_prop,samples,burn_in)

%% codes for the simulation of the posterior distribution of the parameter
%% beta of a probit model, by means of random walk MH algorithm, proposal
%% for beta is multivariate normal centered at current state and covariance
%% matrix tau*V_prop

%% prior on beta multivariate normal (0,v*I)

%% beta_draw is the matrix of the draws for beta, 


%% data is a matrix having as first column the values taken by the response
%% and the remaining columns forming the design matrix 

SAMPLE_RATE=1;

[n, p]=size(X);

% create store variables
beta_draw=zeros(samples,p);

beta_iniz=zeros(p,1);
beta=beta_iniz;
B_prior=v*eye(p);
B_prior_inv=inv(B_prior);

% set counting variable for the number of iterations
no_samples=0;
its=0;


while no_samples < samples 

  its = its+1;
     
  beta_prop=mvrn(beta,tau*V_prop,1)';
  
  % computation of the (log) posterior at the current and the proposal value
  post_current=Y.*log(normcdf(X*beta))+(ones(n,1)-Y).*log((1-normcdf(X*beta)));
  post_current=sum(post_current)-0.5*beta'*B_prior_inv*beta;
  post_prop=Y.*log(normcdf(X*beta_prop))+(ones(n,1)-Y).*log((1-normcdf(X*beta_prop)));
  post_prop=sum(post_current)-0.5*beta_prop'*B_prior_inv*beta_prop;
  alpha_ratio=exp(post_prop-post_current);
  alpha=min(1,alpha_ratio);
  if rand<alpha
      beta=beta_prop;
    
  end
     
     
     
   if its > burn_in && rem(its,SAMPLE_RATE)==0

     no_samples = no_samples + 1;
     beta_draw(no_samples,:)=beta;
  
   
     
   end
  
  
  
  
end





