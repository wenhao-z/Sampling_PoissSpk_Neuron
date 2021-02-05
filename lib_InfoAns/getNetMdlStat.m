function NetStat = getNetMdlStat(nSpk, popVec, parsMdl)
% Calculate the mean and covariance of population vector decoded results
% from neuronal responses r and recurrent input ufwd.

% Note that popVec and nSpk are cells
% with each element the results in a single trial under a parameters set

% Wen-Hao Zhang, July 19, 2019
% wenhao.zhang@pitt.edu
% University of Pittsburgh


%% Statistics of decoder's output (the position on the ring manifold)
DecodeRes  = popVectorDecoder(popVec, parsMdl);
meanSample = mean(DecodeRes, 2);
covSample  = cov(DecodeRes');

% Calculate the covariance of posterior p(s,z|x).
% The reason of doing this is that once I use spiking ufwd, the recorded
% distribution will become p(s,z|s_ext) instead of p(s,z|x). Thus I find
% the covariance of posterior in a parametric way.
if parsMdl.bSample_ufwd && (covSample(4,4) ~= 0)
    covPostSample = covSample(1:2,1:2) - covSample(1,4)^2/ covSample(4,4);
else
    covPostSample = covSample(1:2,1:2);
end

% Fold analyzed results into a struct
NetStat.meanSample = meanSample;
NetStat.covSample = covSample;
NetStat.covPostSample = covPostSample;

%% Statistics of neuronal activity

% Mean firing rate across neurons and across trials
rateAvg = squeeze(mean(nSpk(:)))/parsMdl.tTrial*1e3;

% Correlation coefficient of spike count
corr_r = corr(nSpk');
corr_r = corr_r - diag(diag(corr_r));
corrAvg = mean(corr_r(~isnan(corr_r)), 'all');

ratePop = squeeze(mean(nSpk,2))/parsMdl.tTrial*1e3;

% Get the width and height of population firing rate
% [~, tuneWidth, rateHeight] = getPopRateMetric(ratePop);
tuneParams = lsFitTuneFunc(ratePop, parsMdl); % [Height, posi, Width, Bias]

% Fold analyzed results into a struct
NetStat.ratePop     = ratePop;
NetStat.rateAvg     = rateAvg;
NetStat.corrAvg     = corrAvg;
NetStat.rateHeight  = tuneParams(1);
NetStat.ratePosi    = tuneParams(2);
NetStat.tuneWidth   = tuneParams(3);
NetStat.rateOffset  = tuneParams(4);

end