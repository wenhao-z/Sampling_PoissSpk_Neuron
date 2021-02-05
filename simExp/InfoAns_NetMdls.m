% Demo of information analysis of the network model.

% Wen-Hao Zhang
% July 7, 2019
% University of Pittsburgh

if ~exist('Path_RootDir', 'var')
    setWorkPath;
end
addpath(fullfile(Path_RootDir, 'linearHawkesProcess'));

%% Parameters of the model
parsHawkesNet;

parsMdl.tauIsynDecay = 2;  % Decaying time constant for synaptic input. unit: ms
parsMdl.dt = 0.5; % Simulation time step. unit: ms.
parsMdl.jxe = [0, 0.1, 0.2, 0.5, 1:5]/100; % E synaptic strength.
parsMdl.ratiojie = 5; % Ratio between I and E synaptic strength.
parsMdl.bSample_ufwd = 1; % 1: Generating Poisson input at every time step.
%                           0: Freezed feedforward input.

% Parameters for the test
parsMdl.Posi = 0;  % Input feature, i.e., position on the feature subspace
parsMdl.dPosi = 2; % The difference between a pair of inputs to compute Fisher information
parsMdl.tLen = 5e2*1e3; % The length of simulation. unit: ms
parsMdl.tTrial = 0.2*1e3; % The length of a trial; unit: ms
parsMdl.nTrials = parsMdl.tLen / parsMdl.tTrial; % Number of trials
parsMdl.tStat = 0;

% Parameter of the world
parsMdl.UrecWorld = 30;

% Generate grid of parameters
% Note that every parGrid uses the same random seed.
[parGrid, dimPar] = paramGrid(parsMdl);

%% Simulate the network

% Initialize data structure
tSpkArray = cell([2, size(parGrid)]); % 1st dim: stim. position.
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
    tSpkArray(:, iterPar) = outSet.tSpkArray; % Spike timing
    popVecArray(:, iterPar) = outSet.popVecArray;
    if netpars.bSample_ufwd
        tSpkfwdArray(:, iterPar) = outSet.tSpkfwdArray;
    end
end
clear outSet netpars

%% Partition network's outputs into many segments, with each a trial
nSpk = cellfun(@(tSpk) tSpk2nSpk(tSpk, parsMdl.tTrial, parsMdl), tSpkArray, 'uniformout', 0);
if parsMdl.bSample_ufwd
    nSpkfwd = cellfun(@(tSpk) tSpk2nSpk(tSpk, parsMdl.tTrial, parsMdl), tSpkfwdArray, 'uniformout', 0);
end

subsTrialAvg = [ repmat(1:4, 1, parsMdl.tLen/parsMdl.dt); ...
    kron(1:parsMdl.tLen/parsMdl.tTrial, ones(1, round(4*parsMdl.tTrial/parsMdl.dt)))];
popVec = cellfun(@(x) accumarray(subsTrialAvg.', x(:), []), popVecArray, 'uniformout', 0);

clear subsTrialAvg
%% Information-theoretic analysis

[~, parsMdl.covPosterior] = getPosteriorHierMdl(parsMdl);
parsMdl.meanPosterior = meanPosterior{1}; % Only use Posi(1) to calculate mutual information

% The minimal firing rate to include a cell into analysis
minRateAns = 0; % unit: hz
% Whether using bootstrap to estimate Fisher information to get the error bar
bBootStrap = false;

% Perform information analysis
[InfoAnsRes, NetStat] = InfoTheoAns_BatchFunc(nSpk, popVec, parsMdl, 'minRateAns', ...
    minRateAns, 'bBootStrap', bBootStrap);
if parsMdl.bSample_ufwd
    [InfoAnsFwdRes, NetStatFwd] = InfoTheoAns_BatchFunc(nSpkfwd, popVec, parsMdl, 'minRateAns', ...
        minRateAns, 'bBootStrap', bBootStrap);
end
clear nSpk popVec nSpkfwd

%% Plot the results

figure;
for iter = 1: 4
    hAxe(iter) = subplot(2,2,iter);
    axis square
    hold on
end

% Find the name of the varying model parameters
nameVar = {dimPar.namePar};
IdxPosiDim = cellfun(@(x) strcmp(x, 'Posi'), nameVar);
nameVar(IdxPosiDim) = [];
if length(nameVar) > 1 && (flagModel == 3)
    nameVar = 'JrcRatio';
end
nameVar = nameVar{1};
clear IdxPosiDim

cSpec = lines(3);
axes(hAxe(1))
% Stimulus information in neuronal response
plot(parsMdl.(nameVar), InfoAnsRes.FisherInfo_sext, 'color', cSpec(1,:))
hold on
plot(parsMdl.(nameVar), InfoAnsRes.FisherInfo_sext_theory, 'color', cSpec(2,:))
plot(parsMdl.(nameVar), InfoAnsRes.FisherInfo_sext_theorySimp, 'color', cSpec(3,:))

% Stimulus information in feedforward input
plot(parsMdl.(nameVar), InfoAnsFwdRes.FisherInfo_sext, '--', 'color', cSpec(1,:))
hold on
plot(parsMdl.(nameVar), InfoAnsFwdRes.FisherInfo_sext_theory, '--', 'color', cSpec(2,:))
plot(parsMdl.(nameVar), InfoAnsFwdRes.FisherInfo_sext_theorySimp, '--', 'color', cSpec(3,:))


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

axes(hAxe(4))
% plot(parsMdl.(nameVar), KLDiv(:,1))
yyaxis left
plot(parsMdl.(nameVar), InfoAnsRes.XEnt)
ylabel('Crosss entropy')

yyaxis right
plot(parsMdl.(nameVar), InfoAnsRes.MutualInfo/parsMdl.tTrial*1e3)
[~, IdxMax] = max(InfoAnsRes.MutualInfo);
plot(parsMdl.(nameVar)(IdxMax), InfoAnsRes.MutualInfo(IdxMax)/parsMdl.tTrial*1e3, 'o')
plot(parsMdl.(nameVar), InfoAnsRes.MutualInfo_UpBound/parsMdl.tTrial*1e3, '--')
% ylabel('KL Divergence')
ylabel('Mutual Info. (bit/trial.)')
xlabel(nameVar)

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
