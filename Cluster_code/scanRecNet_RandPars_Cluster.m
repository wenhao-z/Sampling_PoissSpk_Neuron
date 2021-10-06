% Compare the sampling distribution of network with the posterior under
% different input and/or network parameters.

% Wen-Hao Zhang
% July 4, 2021
% University of Chicago

if ~exist('Path_RootDir', 'var')
    setWorkPath;
end
addpath(fullfile(Path_RootDir, 'linearHawkesProcess'));

%% Parameters of the model
parsHawkesNet;

parsMdl.dt = 0.1;
parsMdl.bSample_ufwd = 0;

% Input parameters
parsMdl.tLen = 102*1e3; % unit: ms
parsMdl.tStat = 2*1e3;
parsMdl.tBin = 20; % Decoding time window. unit: ms
% parsMdl.FanoFactorIntVar = 0.7;

nMCSim = 1e3; % Number of Monte Carlo simulation

% Compute the dependent parameters
parsMdl = getDependentPars_HawkesNet(parsMdl);

%% Simulate the network
NetStat = struct('tSample', [], ...
    'meanSample', [], ...
    'covSample', [], ...
    'PreMat_LH', []);
NetStat = repmat(NetStat, nMCSim, 1);

parGrid = struct('Ufwd', [], 'jxe', []);
parGrid = repmat(parGrid, nMCSim, 1);

tStart = clock;
parpool(12);
parfor iterPar = 1: nMCSim
    fprintf('Progress: %d/%d\n', iterPar, nMCSim);
    
    % Load the parameter
    netpar = parsMdl;
    
    % Randomly set the feedforward input parameters and coupling weight
    rng(sum(clock)*100); % I need to randomly set the random seed. The simNet code will set the net random seed.
    
    % Input parameters
    netpar.Ufwd = 50 * rand(1);
    % netpar.jxe = 1e-3 + 1e-2 * rand(1);
    netpar.jxe = 1e-4 + 5e-3 * rand(1);
    
    % Record the randomized input and network parameters
    parGrid(iterPar).Ufwd = netpar.Ufwd;
    parGrid(iterPar).ratiojrprc = netpar.jxe;

    % Generate a sample of feedforward spiking input
    ratefwd = makeRateFwd(netpar.Posi, netpar); % Unit: firing probability in a time bin
    ratefwd = [ratefwd; ...
        netpar.ji0 * sum(ratefwd)/netpar.Ni*ones(netpar.Ni,1)];
    
    % Get the precision of the likelihood
    PreMat_LH = sum(ratefwd(1:netpar.Ne,:),1)/ netpar.TunWidth^2 /2 ...
        * netpar.tBin/netpar.dt; % Precision of the likelihood in time window tBin
    NetStat(iterPar).PreMat_LH = [PreMat_LH, 0; 0, 0];
    
    % Simulate the network
    outSet = simHawkesNet(ratefwd, netpar);
    
    % Compute the statistics of network's samples    
    [tSample, ~, meanSample, covSample] = popVectorDecoder(outSet.popVec(1:2,:), netpar);
    
    NetStat(iterPar).tSample = tSample;
    NetStat(iterPar).meanSample = meanSample;
    NetStat(iterPar).covSample = covSample;
end
tEnd = clock;
clear outSet netpar

%% Compute the statistics of samples and posterior distribution
preSample = arrayfun(@(S) inv(S.covSample(1:2,1:2)), NetStat, 'uniformout', 0);
preSample = cell2mat(shiftdim(preSample,-2));
preSample = squeeze(preSample);

preSamplePred = reshape([NetStat.PreMat_LH], [2,2, size(parGrid)]);

% Numerically find the prior precision stored in the network by comparing
% posterior and likelihood
prePrior = arrayfun(@(x) ...
    findPriorPrecision(parsMdl.Posi * ones(2,1), x.covSample, parsMdl.Posi * ones(2,1), x.PreMat_LH), ...
    NetStat);
% prePrior = -squeeze(preSample(1,2,:));
% prePrior = squeeze(preSample(2,2,:));
prePrior = reshape(prePrior, size(NetStat));
preSamplePred = preSamplePred + shiftdim(prePrior,-2) .* repmat([1,-1;-1,1], [1,1, size(prePrior)]);

%% Save

savePath = fullfile(Path_RootDir, 'Data', 'HawkesNet');

str = datestr(now, 'yymmddHHMM');
fileName = ['SingleRecNet_RandPars_', str(1:6), '_', str(7:end) '.mat'];

save(fullfile(savePath, fileName), '-v7.3')

