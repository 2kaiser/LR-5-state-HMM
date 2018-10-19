function [ out ] = B( X, mu, sigma )
%Posterior porbability calculation 
%   this functino calculates the matrix B where each 
%   (i,t)th element is the the probability of xt given
%   that we are in state i.

%   INPUTS:
%       X -- is the observation vector of dimension d by t where t is the 
%       number of timestamps and d is the dimension of the feature vector
%       mu -- an X by d matrix of the means for each state of the hmm where
%       X is the number of states
%       sigma -- a X by d by d matirx with the variances for each state 
invsigma = inv(sigma);

out = X;
ObsLength = len(X);
dimen = size(mu,2); %number of 
d = size(mu,1); %the number of states
temp = 0;
for i = 1:ObsLength
    for t = 1:d %number of states
        temp = 0;
        for d = 1:dimen %dimension of feature vector
            temp = 1/(2*pi*invsigma(d,d,i))^(1/2) * exp((-1/2)*(X[d,t]-mu[d,i])^2/invsigma(d,d,i));
        out[i,t] = temp;
   
            

end

