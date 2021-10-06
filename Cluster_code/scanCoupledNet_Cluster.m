% Scan the network parameter to test the robustness of sampling in a coupled network.

% Wen-Hao Zhang, June 15, 2021
% University of Chicago

setWorkPath;
addpath(fullfile(Path_RootDir, 'linearHawkesProcess'));

%% Parameters of the model
parsCoupledHawkesNet;

parsMdl.dt = 0.1;
parsMdl.bSample_ufwd = 0;
parsMdl.jxe = 1e-2;
parsMdl.bCutRecConns = 1; % Cut off all recurrent connections to simplify theoretical analysis
parsMdl.bShareInhPool = 0; % Whether two networks share the same inhibition pool

% Input parameters
parsMdl.Posi = 10*[-1; 1];
parsMdl.tLen = 102*1e3; % unit: ms
parsMdl.tStat = 2*1e3;
parsMdl.tBin = 20; % Decoding time window. unit: ms

% Adjustable parameters
NUfwd = 50;
parsMdl.Ufwd = 50 * rand(2, NUfwd);
% parsMdl.Ufwd = [0:3, 5:5:50]; % Peak firing rate of feedforward inputs, unit: Hz
% parsMdl.Ufwd = [parsMdl.Ufwd; 25* ones(size(parsMdl.Ufwd ))];
parsMdl.ratiojrprc = 0:0.5:5;

% Generate grid of parameters
% Note that every parGrid uses the same random seed.
[parGrid, dimPar] = paramGrid(parsMdl);

% Compute the dependent parameters
parGrid = arrayfun(@(x) getDependentPars_HawkesNet(x), parGrid);

%% Simulation
NetStat = struct('tSample', [], ...
    'meanSample', [], ...
    'meanSamplePred', [], ...
    'covSample', [], ...
    'covCondMean', [], ...
    'PreMat_LH', [], ...
    'ratePop', [], ...
    'rateHeight', []);
NetStat = repmat(NetStat, size(parGrid));

tStart = clock;
parpool(12);
parfor iterPar = 1: numel(parGrid)
    fprintf('Progress: %d/%d\n', iterPar, numel(parGrid));
    
    netpar = parGrid(iterPar);
    
    % Generate a sample of feedforward spiking input
    ratefwd = makeRateFwd(netpar.Posi, netpar);
    ratefwd = [ratefwd; netpar.ji0 * sum(ratefwd,1)./netpar.Ni.*ones(netpar.Ni,1)];
    
    % Get the precision of the likelihood
    preLH = sum(ratefwd(1:netpar.Ne,:),1)/ netpar.TunWidth^2 /2 ...
        * netpar.tBin/netpar.dt; % Precision of the likelihood in time window tBin
    NetStat(iterPar).PreMat_LH = diag(preLH);
    
    % Simulate the network
    outSet = simCoupledHawkesNet(ratefwd(:), netpar);
    
    % Compute the statistics of network's samples
    % Spike count in of each trial
    tEdge = [netpar.tStat+netpar.dt, netpar.tLen];
    neuronEdge = [0.5: netpar.Ne + 0.5, netpar.Ncells+ (0.5: netpar.Ne + 0.5)];
    nSpk = histcounts2(outSet.tSpk(1,:)', outSet.tSpk(2,:)', neuronEdge, tEdge);
    nSpk(netpar.Ne+1) = [];
    
    NetStat(iterPar) = getNetMdlStat(nSpk, outSet.popVec, netpar, NetStat(iterPar));
end

tEnd = clock;

%% Theoretical prediction of sampling distribution
preSample = arrayfun(@(x) inv(x.covSample), NetStat, 'uniformout', 0);
preSample = cell2mat(shiftdim(preSample,-2));

% Compute the recurrent input strength to predict the prior
% Note: only work for two coupled networks
prePriorPred = arrayfun(@(x) mean(x.rateHeight) * [1,-1;-1,1], NetStat, 'uniformout', 0);
prePriorPred = cell2mat(shiftdim(prePriorPred,-2));
prePriorPred = prePriorPred .* shiftdim(parsMdl.ratiojrprc,-1);
prePriorPred = prePriorPred * sqrt(2*pi)*parsMdl.Ne * parsMdl.jxe/sqrt(parsMdl.Ncells)/parsMdl.TunWidth;
prePriorPred = prePriorPred * parsMdl.tBin * parsMdl.dt/1e3; % The precision in the time window tBin

% Numerically find the prior precision stored in the network by comparing
% posterior and likelihood
prePrior = arrayfun(@(x) findPriorPrecision(x.meanSample, x.covSample, parsMdl.Posi, x.PreMat_LH), ...
    NetStat);

%% Save

savePath = fullfile(Path_RootDir, 'Data', 'HawkesNet');

str = datestr(now, 'yymmddHHMM');
fileName = ['CoupledNet_', str(1:6), '_', str(7:end) '.mat'];

save(fullfile(savePath, fileName), '-v7.3')

