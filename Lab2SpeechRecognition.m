

%%
%
%loading in the training data 
%

dg_asr1 = load('dg_asr1.fea');
dg_asr2 = load('dg_asr2.fea');
dg_asr3 = load('dg_asr3.fea');
dg_asr4 = load('dg_asr4.fea');
dg_asr5 = load('dg_asr5.fea');



%
%initialize parameters: A, pi, mu, and sigma
%


%
%A is given to us 
%

A = [0.8 0.2 0 0 0; 0 0.8 0.2 0 0; 0 0 0.8 0.2 0; 0 0 0 0.8 0.2;
0 0 0 0 1];

% 
%assume uniform distribution across words for (pi)i
%

pi = [1/5 1/5 1/5 1/5 1/5];

%
%to obtain each mean vector we average four utterances of the same word
%along the dimension of the mfcc features, and concatenated them.
%We need one mean vector per state
%So we will have 5 1 x 14 dimension mean vectors (mu)i 
%and finally a 5 X 14 mu vectors in the variable mu
%It doesn't matter what we initialize the mean as since we do EW with Baum
%Welch


mu1 = mean(dg_asr1,1);
mu2 = mean(dg_asr2,1);
mu3 = mean(dg_asr3,1);
mu4 = mean(dg_asr4,1);
mu5 = mean(dg_asr5,1);


mu = cat(1,mu1,mu2,mu3,mu4,mu5);


%
%compute the covariance for each feature vector.  We need one for each
%state
%So we will end up having 5 14 X 14 covariance vectors
%

cov1 = cov(dg_asr1);
cov2 = cov(dg_asr2);
cov3 = cov(dg_asr3);
cov4 = cov(dg_asr4);
cov5 = cov(dg_asr5);
cov = cat(3,cov1,cov2,cov3,cov4,cov5);

%
%setting up the posterior probability bi(xt) = p(xt | qt = i)
%


%

%%