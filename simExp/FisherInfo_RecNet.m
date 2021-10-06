% Estimate the Fisher information of stimulus in the neuronal responses of a single 
% recurrent network
% Some simplifications is used in this Hawkess process:
% 1). No temporal kernel

% Wen-Hao Zhang
% July 22, 2021
% University of Chicago

if ~exist('Path_RootDir', 'var')
    setWorkPath;
end
addpath(fullfile(Path_RootDir, 'linearHawkesProcess'));

%% Parameters of the model

parsHawkesNet;

parsMdl.dt = 0.1;

% parsMdl.jxe = [0, 0.1, 0.2, 0.5, 1:5]/100;
parsMdl.jxe = (0:0.1:1)/100;
parsMdl.ratiojie = 5;
parsMdl.bSample_ufwd = 1;

% Parameters for the test
parsMdl.Posi = 0;
parsMdl.dPosi = 2;
parsMdl.tLen = 50*1e3; % unit: ms
parsMdl.tTrial = 0.2*1e3; % unit: ms
parsMdl.nTrials = parsMdl.tLen / parsMdl.tTrial;
parsMdl.tStat = 0;

% Parameter of the world
parsMdl.UrecWorld = 20;

% Generate grid of parameters
% Note that every parGrid uses the same random seed.
[parGrid, dimPar] = paramGrid(parsMdl);

%% Simulate the network

% Initialize data structure
tSpkArray     = cell([2, size(parGrid)]); % 1st dim: stim. position.
if parsMdl.bSample_ufwd
    tSpkfwdArray = cell([2, size(parGrid)]);
end

popVecArray   = cell([2, size(parGrid)]);
meanPosterior = cell([2, size(parGrid)]); % [(s,z), dim of parGrid];

parfor iterPar = 1: numel(parGrid)
    fprintf('Progress: %d/%d\n', iterPar, numel(parGrid));
    
    % Load the parameter
    netpars = parGrid(iterPar);
    
    % Simulate a network model to generate spiking responses
    outSet = simNetMdl_FisherInfo(netpars);
    
    meanPosterior(:, iterPar) = outSet.meanPosterior;
    tSpkArray(:, iterPar) = outSet.tSpkArray;
    popVecArray(:, iterPar) = outSet.popVecArray;
    if netpars.bSample_ufwd
        tSpkfwdArray(:, iterPar) = outSet.tSpkfwdArray;
    end
end
clear outSet netpars

%% Partition network's outputs into many trials

% Convert spike timing into spike count in a trial
nSpk = cellfun(@(tSpk) tSpk2nSpk(tSpk, parsMdl.tTrial, parsMdl), tSpkArray, 'uniformout', 0);
if parsMdl.bSample_ufwd
    nSpkfwd = cellfun(@(tSpk) tSpk2nSpk(tSpk, parsMdl.tTrial, parsMdl), tSpkfwdArray, 'uniformout', 0);
end

% Partition linear decoder output into many trials
subsTrialAvg = [ repmat(1:4, 1, parsMdl.tLen/parsMdl.dt); ...
    kron(1:parsMdl.tLen/parsMdl.tTrial, ones(1, round(4*parsMdl.tTrial/parsMdl.dt)))];
popVec = cellfun(@(x) accumarray(subsTrialAvg.', x(:), []), popVecArray, 'uniformout', 0);
clear subsTrialAvg

% Get the statistics of neuronal responses
NetStat = cellfun(@(nspk, popvec) getNetMdlStat(nspk, popvec, parsMdl), nSpk, popVec);

%% Estimate Fisher information

% The minimal firing rate to include a cell into analysis
minRateAns = 0; % unit: hz
% Number of bootstrap to estimate the std. of Fisher information. 
% 0: doesn't use bootstrap
nBootStrap = 0;
% nBootStrap = 50;

% Estimate Fisher information
[InfoAnsRes.FisherInfo_sext, InfoAnsRes.stdFisherInfo_sext] = ...
    cellfun(@(n1,n2) fisherInfo_biasCorrect(n1, n2, parsMdl.dPosi*[-1,1]/2, parsMdl, ...
        'minRateAns', minRateAns, 'nBootStrap', nBootStrap), ...
        nSpk(1,:), nSpk(2,:));    
    
% ------------------------------------------------------------
% Theoretical prediction of linear Fisher information
[InfoAnsRes.FisherInfo_sext_theory, InfoAnsRes.FisherInfo_sint_theory] = arrayfun(@(S) ...
    fisherInfo_Theory(S.rateHeight, S.tuneWidth, S.rateOffset, S.covSample(3,3), parsMdl), ...
    NetStat(1,:));
% A simple theoretical prediction of linear Fisher information
[InfoAnsRes.FisherInfo_sext_theorySimp, InfoAnsRes.FisherInfo_sint_theorySimp] = arrayfun(@(S) ...
    fisherInfo_TheorySimple(S.rateAvg, parsMdl.Ne, S.tuneWidth, S.covSample(3,3)), ...
    NetStat(1,:));

%% Plot the results
figure;
for iter = 1:3
    hAxe(iter) = subplot(1,3,iter);
    axis square
    hold on
end

% Find the name of the varying model parameters
nameVar = {dimPar.namePar};
IdxPosiDim = cellfun(@(x) strcmp(x, 'Posi'), nameVar);
nameVar(IdxPosiDim) = [];
if length(nameVar) > 1    
    nameVar = 'jxe';
end
% nameVar = nameVar{1};
clear IdxPosiDim

cSpec = lines(3);
axes(hAxe(1))
% Stimulus information in neuronal response
plot(parsMdl.(nameVar), InfoAnsRes.FI_sext, 'color', cSpec(1,:))
hold on
plot(parsMdl.(nameVar), InfoAnsRes.FI_sext_theory, 'color', cSpec(2,:))
plot(parsMdl.(nameVar), InfoAnsRes.FI_sext_theorySimp, 'color', cSpec(3,:))

% Stimulus information in feedforward input
plot(parsMdl.(nameVar), InfoAnsRes.FI_sext_ufwd, '--', 'color', cSpec(1,:))
% hold on
% plot(parsMdl.(nameVar), InfoAnsFwdRes.FI_sext_theory, '--', 'color', cSpec(2,:))
% plot(parsMdl.(nameVar), InfoAnsFwdRes.FI_sext_theorySimp, '--', 'color', cSpec(3,:))


ylabel('Linear Fisher Info. ((deg^2 sec)^{-1})')
legend('Sim.', 'Theory')
title(sprintf('tTrial=%1dms', parsMdl.tTrial));

axes(hAxe(2))
plot(parsMdl.(nameVar), arrayfun(@(S) S.covSample(1,1), NetStat(1,:)))
plot(parsMdl.(nameVar), arrayfun(@(S) S.covSample(3,3), NetStat(1,:)))
plot(parsMdl.(nameVar)(2:end), arrayfun(@(S) S.covSample(2,2), NetStat(1,2:end)));
legend('V(s)', 'Diff. corr. (V(\mu_s))', 'V(z)', 'location', 'best')
ylabel('Variance')
title('Linear decoder')

axes(hAxe(3))
yyaxis('left')
% plot(parsMdl.(nameVar), NetStat.rateAvg)
plot(parsMdl.(nameVar), [NetStat(1,:).rateHeight])
hold on
plot(parsMdl.(nameVar)(IdxMax), [NetStat(1,IdxMax).rateHeight], 'o')
ylabel('Bump height (Hz)')

yyaxis('right')
plot(parsMdl.(nameVar), [NetStat(1,:).corrAvg])
ylabel('Mean corr. coef.')

linkaxes(hAxe, 'x')
set(hAxe, 'xlim', parsMdl.(nameVar)([1,end]))
