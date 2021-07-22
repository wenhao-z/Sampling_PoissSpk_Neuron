function [mu, Sigma] = getPosteriorHierMdl(parsMdl)
% Calculate the mean and covariance of posterior in the hierarhical model
% considered in my Cosyne 2019 study.
% Wen-Hao Zhang, July 9, 2019
% University of Pittsburgh

mu = parsMdl.Posi;
    
Sigma = 1./parsMdl.Ufwd * ones(2,2); % The cov. from likeihood function 
% if parsMdl.bSample_ufwd
%     % When bSample_ufwd is true, it means the external stimulus is fixed,
%     % instead of the feedforward input. 
%     % Thus, the criteria of the posterior distribution should marginalize
%     % x.
%     Sigma = 2 * Sigma;
% end

% The cov. from prior
if parsMdl.UrecWorld == 0
    % The var. of global context z satisfies a uniform distribution when
    % UrecWorld is zero
    Sigma(2,2) = Sigma(2,2) + parsMdl.width^2/3;     
else
    Sigma(2,2) = Sigma(2,2) + 1./parsMdl.UrecWorld;
end
Sigma = Sigma * parsMdl.TunWidth/ (sqrt(2*pi) * parsMdl.rho); % This line has a bug which is inconsistent with the width of feedforward input in makeRateFwd.m
% Sigma = Sigma * sqrt(2)*parsMdl.TunWidth/ (sqrt(2*pi) * parsMdl.rho);
Sigma = Sigma ./ parsMdl.tTrial*1e3; % scaled to the duration of each trial