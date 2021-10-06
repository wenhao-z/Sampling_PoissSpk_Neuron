function [XEnt, MutualInfo, MutualInfo_UpBound] = getMutualInfo(mu0, Cov0, mu, Cov, L, CovPrior)
% Calculate the cross entropy between two normal distributions
% Wen-Hao Zhang, June 27, 2021
% University of Chicago

% mu0 and Cov0: the mean and covariance of Posterior
% mu and Cov: the mean and covariance of sampling distribution
% CovPrior is a scalar variable!


if ~ exist('CovPrior', 'var')
    CovPrior = Cov0(2,2) - Cov0(1,1);
elseif length(CovPrior) ~= 1
   error('CovPrior should be a scalar!') 
end

% Cross entropy - E_q[ln p]
dimCov = length(Cov0);
XEnt = dimCov*log(2*pi) ...
    + log(det(Cov)) + trace(Cov \ Cov0 ) ...
    + (mu - mu0)' / Cov * (mu - mu0);
XEnt = XEnt /2;

% Entropy of the prior distribution, i.e., p(s,z)
% - E_prior [ ln prior ]

EntPrior = 1 + log(2*pi) + log(CovPrior);
EntPrior = EntPrior/2 + log(L);

MutualInfo = (EntPrior - XEnt)/ log(2); % Unit: bit


MutualInfo_UpBound = EntPrior - (log(2*pi)+ 1) - log(det(Cov0))/2;
MutualInfo_UpBound = MutualInfo_UpBound/ log(2); % Unit: bit

