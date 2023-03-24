function [beta_means,beta_vars,sigma,lambdas] = BMA(xraw,y,nsave,nburn)

% ===========================| USER INPUT |================================
% Gibbs related preliminaries
ntot = nsave + nburn;   % Total draws

%prior = 2;            % 1. g = 1/n
                       % 2. g = 1/k^2
     
% ===========================| LOAD DATA |=================================

T = size(xraw,1); % time series observations
k = size(xraw,2); % maximum nubmer of predictors

% molddraw is initialized to contain all explanatory variables
% with t-stats greater than .5
molddraw = zeros(k,1);

% Make up the matrix of explanatory variables for model
xold = [ones(T,1) xraw(:,molddraw>0)];
kold = sum(molddraw)+1;

% Specify g0 for the g-prior
if T <= k
    g = 1/k^2;
else
    g = 1/T;
end

yty = (y-mean(y))'*(y-mean(y));
xtxinv=inv(xold'*xold);
ymy = y'*y - y'*xold*xtxinv*xold'*y;
g1=g/(g+1);
g2=1/(g+1);
lprobold = .5*kold*log(g1) -.5*(T-1)*log(g2*ymy + g1*yty);
mstart=molddraw;
lprobstart=lprobold;
inccount=zeros(k,1);
msize=0;

%I am keeping records for top 10 drawn models
%Here initialize this to initial model
top10mod = [molddraw molddraw molddraw molddraw molddraw ...
       molddraw molddraw molddraw molddraw molddraw];
lprobtop10=lprobold*ones(10,1);
top10count=zeros(10,1);

%calculate first and second moment of all coefficients
%Initialize them here
b1mo=zeros(k,1);
b2mo=zeros(k,1);
sigma_draw = zeros(nsave,1);
%==========================================================================
%==========================| GIBBS SAMPLER |===============================
tic;
fprintf('Now you are running BMA')
fprintf('\n')
fprintf('Iteration 0000')

for irep=1:ntot
    % Print iterations   
    if mod(irep,500) == 0
        fprintf('%c%c%c%c%c%c%c%c%c%c%c%c%c%c%s%4d',8,8,8,8,8,8,8,8,8,8,8,8,8,8,'Iteration ',irep)
    end
    
    %choose at random one of the k potential explanatory variables  
    %if it is already in the model, delete it else add it
    %Based on this, make up candidate model
    indch=round(k*rand);
    xnew=xold;
    mnewdraw=molddraw;
    if indch>0
        if molddraw(indch,1)==1
            isum=0; 
            for i=1:indch
                isum=isum+molddraw(i,1);
            end
            xnew = [xold(:,1:isum) xold(:,isum+2:kold)];
            mnewdraw(indch,1)=0;    
        else
            isum=0;
            for i=1:indch
                isum=isum+molddraw(i,1);
            end
            xnew = [xold(:,1:isum+1) xraw(:,indch) xold(:,isum+2:kold)];
            mnewdraw(indch,1)=1;
        end
    end
    
    knew=sum(mnewdraw)+1;
    xtxinv=inv(xnew'*xnew);
    ymy = y'*y - y'*xnew*xtxinv*xnew'*y;
    lprobnew = .5*knew*log(g1) -.5*(T-1)*log(g2*ymy + g1*yty); 
    
    %Now decide whether to accept candidate draw
    if log(rand) < (lprobnew - lprobold)
        xold=xnew;
        lprobold=lprobnew;
        molddraw=mnewdraw;
        kold=knew;
    end
    
    if irep>nburn
        %If new drawn model better than current top 10, add it to list  
        for i=1:10
            if lprobold>=lprobtop10(i,1)
                if sum(abs(molddraw - top10mod(:,i)))<.09
                    break
                end
                if i<10
                    lprobtop10(i+1:10,1)=lprobtop10(i:9,1);
                    top10mod(:,i+1:10) = top10mod(:,i:9);
                    top10count(i+1:10,1)=top10count(i:9,1);
                end
                lprobtop10(i,1)=lprobold;
                top10mod(:,i)=molddraw;
                top10count(i,1)=0;
                break
            end
        end
        
        for i=1:10
            temp1=sum(abs(molddraw-top10mod(:,i)));       
            if temp1<.01
                top10count(i,1)=top10count(i,1)+1;
                break
            end
        end        
        inccount = inccount + molddraw;
        msize=msize + kold;
        %calculating posterior properties of coefficients means
        %we have to write out full posterior
        Q1inv = (1+g)*xold'*xold;
        Q0inv  =g*xold'*xold;       
        Q1=inv(Q1inv);
        b1= Q1*xold'*y;
        vs2 = (y-xold*b1)'*(y-xold*b1) + b1'*Q0inv*b1;
        sigma_draw(irep-nburn,1) = vs2/(T-2);
        bcov = (vs2/(T-2))*Q1;
        %the next bit of this is awkward, needed to find out if variable is 
        %included in the model and, if so, to find out where it is
        summer1=1;
        for i=1:k
            bc=zeros(1,kold);        
            if molddraw(i,1)==1
                summer1=summer1+1;
                bc(1,summer1)=1;
                bmean=bc*b1;
                bvar =bc*bcov*bc';          
                b1mo(i,1)=b1mo(i,1) + bmean;
                b2mo(i,1)=b2mo(i,1) + (bvar+bmean^2);
            end
        end
    end
    
    
end

fprintf('%c%c%c%c%c%c%c%c%c%c%c%c%c%c',8,8,8,8,8,8,8,8,8,8,8,8,8,8)
% fprintf('%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c',8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8)
toc; % Stop timer and print total time

%===========================| END SAMPLER |================================
%==========================================================================
b1mo=b1mo./nsave;
b2mo=b2mo./nsave;
bsd=sqrt(b2mo-b1mo.^2);

beta_means = b1mo;
beta_vars  = bsd;
lambdas=inccount./nsave;
sigma = mean(sigma_draw);