function [XEnt, MutualInfo, MutualInfo_UpBound] = getMutualInfo(mu0, Cov0, mu, Cov, L, CovPrior)
% Calculate the cross entropy between two normal distributions
% Wen-Hao Zhang, June 27, 2019
% University of Pittsburgh

dimCov = length(Cov0);

XEnt = dimCov*log(2*pi) ...
    + log(det(Cov)) + trace(Cov0 / Cov) ...
    + (mu - mu0)' / Cov * (mu - mu0);
XEnt = XEnt /2;

% Entropy of the prior distribution, i.e., p(s,z)
if ~ exist('CovPrior', 'var')
    CovPrior = Cov0(2,2) - Cov0(1,1);
end
EntPrior = 1 + log(2*pi) + log(det(CovPrior));
EntPrior = EntPrior/2 + log(L);

MutualInfo = (EntPrior - XEnt)/ log(2); % Unit: bit


MutualInfo_UpBound = EntPrior - (log(2*pi)+ 1) - log(det(Cov0))/2;
MutualInfo_UpBound = MutualInfo_UpBound/ log(2); % Unit: bit

