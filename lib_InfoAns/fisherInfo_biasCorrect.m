function [Info_sext, stdInfo] = fisherInfo_biasCorrect(nSpk1, nSpk2, Posi, NetPars, varargin)
% Estimate the Fisher information of external stimulus by using bias correlation
% Ref: Kanitscheider, Plos Comp. Biol. 2015

% nSpk: [N, nTrials] array, number of spikes in a trial
% The number of Posi should be 2

% Wen-Hao Zhang, Aug 5, 2019
% wenhao.zhang@pitt.edu


% Get the possible parameters from varargin
if mod(size(varargin,2) , 2) == 1 % odd number input
    error('The varargin input number is wrong!')
else
    for iter = 1: round(size(varargin,2)/2)
        eval([varargin{2*iter-1} '= varargin{2*iter};']);
    end
end
clear iter

% Default parameters 
if ~exist('minRateAns', 'var')
    % The minimal firing rate to include a cell into analylsis
    minRateAns = 0; % unit: hz
end
if ~exist('nBootStrap', 'var')
    nBootStrap = 0;
end

% ------------------------------------------------------------
% Initialization
nTrials = size(nSpk1, 2);
dPosi = Posi(1)-Posi(2);

% ------------------------------------------------------------
% Estimate the Fisher information
% May use bootstrap to estimate the variability of information
Info_sext_bootstrap = zeros(1, nBootStrap+1); 
% Info_sext_bootstrap(1): the Fisher information without randomly draw data in bootstrap

for iter = 1: (nBootStrap+1)
    if (nBootStrap>0) && (iter > 1)
        IdxTrial = datasample(1: nTrials, nTrials);
    else
        % The 1st iteration uses _original data_ without bootstrap even if
        % bootstrap is turned on 
        IdxTrial = 1:nTrials;
    end
    
    % Find neurons firing spikes
    nSpkAvg1 = squeeze(mean(nSpk1(:,IdxTrial), 2)); % Average of spike count across trials.
    nSpkAvg2 = squeeze(mean(nSpk2(:,IdxTrial), 2)); % Average of spike count across trials.
    IdxNeuronAns = (nSpkAvg1> minRateAns*NetPars.tTrial/1e3) & ...
        (nSpkAvg2> minRateAns*NetPars.tTrial/1e3);
    
    drate_dPosi = (nSpkAvg1 - nSpkAvg2)/ dPosi;
    cov_r = (cov(nSpk1(IdxNeuronAns, IdxTrial)') + ...
        cov(nSpk2(IdxNeuronAns, IdxTrial)'))/2;
    
    % Fisher information of external stimulus
    Info_sext = drate_dPosi(IdxNeuronAns)' / cov_r * drate_dPosi(IdxNeuronAns);
    
    % Correct the bias of Fisher information estimation
    Info_sext = Info_sext * ( 1 - (NetPars.Ne+1)/(2*NetPars.nTrials-2) ) ...
        - 2*NetPars.Ne/NetPars.nTrials / dPosi^2;
    
    Info_sext_bootstrap(iter) = Info_sext/NetPars.tTrial*1e3;
end

Info_sext = Info_sext_bootstrap(1);

% Standard deviation of Fisher information estimated from bootstrap
if nBootStrap > 0
    stdInfo = std(Info_sext_bootstrap(2:end));
else
    stdInfo = 0;
end

end