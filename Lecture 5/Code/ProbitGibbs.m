function [beta_draw,zeta_draw]=ProbitGibbs(Y,X,beta_0,B_0,samples,burn_in)

%% codes for the simulation of the posterior distribution of the parameter
%% beta of a probit model, by means of auxiliary variables gibbs sampler
%% see Albert and Chib (1993) Bayesian Analysis of Binary and Polychotomous Response Data (JASA)

%% beta_0 is a column vector

%% beta is the matrix of the draws for beta, zeta_draw is the matrix of the
%% values drawn for the latent variables


%% data is a matrix having as first column the values taken by the response
%% and the remaining columns forming the design matrix
    
SAMPLE_RATE = 1;

[n, p] = size(X);

% create store variables
beta_draw=zeros(samples,p);
zeta_draw=zeros(samples,n);


beta  = mvrn(beta_0,B_0,1)';
zeta  = X*beta;
XtX   = X'*X;
B_inv = inv(B_0);


% set counting variable for the number of iterations
no_samples=0;
its=0;

while no_samples < samples 

  its = its+1;
     P_post=(B_inv+XtX);
     B_post=inv(P_post);
     m_post=B_post*(B_inv*beta_0+X'*zeta);
    
     beta = mvrn(m_post,B_post,1)';
  
     for i=1:n
         if Y(i)==1
             zeta(i)=trunc(X(i,:)*beta,1,1,1);
         else
            zeta(i)=trunc(X(i,:)*beta,1,1,0);
         end
     end
   if its > burn_in && rem(its,SAMPLE_RATE)==0

     no_samples = no_samples + 1;
     beta_draw(no_samples,:)=beta;
     zeta_draw(no_samples,:)=zeta;
     
     
   end
  
  
  
  
end





