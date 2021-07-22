% Demo of the network model with each neurons modeled as a linear Hawkes
% process. The E neurons are arranged on a ring according to their
% preferred periodic stimulus feature, while the I neurons in the network
% are not selective to the stimulus feature.

% There are three variability in this model:
% 1. Poissonian feedforward input.
% 2. Poisson spike generation in the model.
% 3. Poisson-like recurrent variability.

% Wen-Hao Zhang, July 2, 2019
% University of Pittsburgh

%% Parameters of the model
parsHawkesNet;

parsMdl.tauIsynDecay = 2; % Decaying time constant for synaptic input. unit: ms
parsMdl.dt = 0.1; % Simulation time step. unit: ms.
parsMdl.tLen = 200*1e3; % unit: ms 
parsMdl.bSample_ufwd = 1; % 1: Generating Poisson input at every time step.
%                           0: Freezed feedforward input.
parsMdl.jxe = 2e-2; % E synaptic strength.
parsMdl.ratiojie = 5; % Ratio between I and E synaptic strength.
parsMdl.FanoFactorIntVar = 1; % The Fano factor of recurrent interactions.

parsMdl.tTrial = 0.2*1e3; % length of a trial. Unit: sec

% Compute the dependent parameters
parsMdl = getDependentPars_HawkesNet(parsMdl);

%% Simulation

% Generate a sample of feedforward spiking input
ratefwd = makeRateFwd(parsMdl.Posi, parsMdl); % The rate of feedforward input to E neurons
ratefwd = [ratefwd; parsMdl.ji0 * sum(ratefwd)/parsMdl.Ni*ones(parsMdl.Ni,1)]; % Concatenate the input to I neurons

% Simulate the network model
tic
outSet = simHawkesNetDemo(ratefwd, parsMdl);
toc
%%
% Spike count in of each trial
% Partition a long trial neural response into disjoint trials (segments)
tEdge = [0: parsMdl.tTrial : parsMdl.tLen, parsMdl.tLen + parsMdl.tTrial];
neuronEdge = 0.5: parsMdl.Ne + 0.5;
nSpk = histcounts2(outSet.tSpk(1,:)', outSet.tSpk(2,:)', neuronEdge, tEdge);

neuronEdgeI = (0.5: parsMdl.Ni + 0.5) + parsMdl.Ne;
nSpkI = histcounts2(outSet.tSpk(1,:)', outSet.tSpk(2,:)', neuronEdgeI, tEdge);

% Get the network response statistics
subsTrialAvg = [ repmat(1:4, 1, parsMdl.tLen/parsMdl.dt); ...
    kron(1:parsMdl.tLen/parsMdl.tTrial, ones(1, round(4*parsMdl.tTrial/parsMdl.dt)))];
popVec = accumarray(subsTrialAvg.', outSet.popVec(:), []);
NetStat = getNetMdlStat(nSpk, popVec, parsMdl);

% Covariance and correlation of neuronal activity
CovRate = cov(nSpk')/parsMdl.tTrial*1e3;
CorrRate = corr(nSpk');% Mean of decoded results in each trial

% Calculate the noise correlation with difference between two neurons'
% prefered stimulus
[rowSub, colSub] = ind2sub(size(CovRate), 1:numel(CovRate));
diffInd = abs(rowSub - colSub);
diffInd(diffInd>length(CovRate)/2) = diffInd(diffInd>length(CovRate)/2) - length(CovRate);
diffInd = abs(diffInd);

CovRate_diffPrefStim = accumarray(diffInd(:)+1, CovRate(:), [], @mean);
CorrRate_diffPrefStim = accumarray(diffInd(:)+1, CorrRate(:), [], @mean);
diffPrefStim = mean(diff(parsMdl.PrefStim)) * unique(diffInd);

% Analysis of the feedforward input
if parsMdl.bSample_ufwd
    nSpkFwd = histcounts2(outSet.tSpkfwd(1,:)', outSet.tSpkfwd(2,:)', neuronEdge, tEdge);
    Covufwd = cov(nSpkFwd')/parsMdl.tTrial*1e3;
end

% Decoded results
popVec = accumarray(subsTrialAvg.', outSet.popVec(:), []);
DecodeRes = popVectorDecoder(popVec, parsMdl);

% Get the prediction of posterior from firing rate
parsMdlTmp = parsMdl;
parsMdlTmp.UrecWorld = NetStat.rateHeight - parsMdl.Ufwd;
parsMdlTmp.TunWidth = NetStat.tuneWidth;
[~, covPosterior] = getPosteriorHierMdl(parsMdlTmp);

clear rowSub colSub subs diffInd popVec parsMdlTmp subsTrialAvg
%% Plot the results

tLim = 5e2;
tSpk = outSet.tSpk;
tSpk(:, tSpk(2,:)>tLim) = [];

plotSpkRaster(tSpk, parsMdl, 'tBin', 0.01*1e3);
set(gca, 'xlim', [0, tLim])

%%
figure
subplot(3,2,1)
plot(parsMdl.PrefStim, NetStat.ratePop)
hold on;
plot(parsMdl.PrefStim, NetStat.ratePop + sqrt(diag(CovRate)));
plot(parsMdl.PrefStim, NetStat.ratePop - sqrt(diag(CovRate)));
set(gca, 'xlim', parsMdl.PrefStim(end)*[-1,1], 'xtick', -180:90:180)
ylabel('Firing rate (Hz)')

subplot(3,2,2)
plot(parsMdl.PrefStim, diag(CovRate));
ylabel('Cov. of rate')
set(gca, 'xlim', parsMdl.PrefStim(end)*[-1,1], 'xtick', -180:90:180)

subplot(3,2,3)
imagesc(parsMdl.PrefStim, parsMdl.PrefStim, CovRate - diag(diag(CovRate)));
% imagesc(CorrRate - diag(diag(CorrRate)));
set(gca, 'xlim', parsMdl.PrefStim(end)*[-1,1], 'ylim', parsMdl.PrefStim(end)*[-1,1], ...
    'xtick', -180:90:180, 'ytick', -180:90:180)
axis xy
title('Cov. of rate')
axis square

subplot(3,2,4)
yyaxis left
plot(diffPrefStim(2:end), CovRate_diffPrefStim(2:end));
ylabel('Cov. of rate')
yyaxis right
plot(diffPrefStim(2:end), CorrRate_diffPrefStim(2:end));
ylabel('Corr. of rate')
set(gca, 'xlim', diffPrefStim([1, end]), 'xtick', 0:90:180)

subplot(3,2,5)
plot(1:parsMdl.Ni, mean(nSpkI,2))
hold on;
plot(1:parsMdl.Ni, mean(nSpkI,2) + std(nSpkI, 0,2));
plot(1:parsMdl.Ni, mean(nSpkI,2) - std(nSpkI, 0,2));
set(gca, 'xlim', [1, parsMdl.Ni], 'xtick', [1, parsMdl.Ni], ...
    'ylim', [0, 15])
ylabel('Firing rate (Hz)')


%%
figure; clf;
hAxe = plotJointMarginalHist(DecodeRes(1,:), DecodeRes(2,:));
axes(hAxe(1))
xlabel('Local feature s')
ylabel('Global feature z')
axis(hAxe(1), 40*[-1,1,-1,1])
set(hAxe(2), 'ylim', [0, 0.15])
set(hAxe(3), 'xlim', [0, 0.15])


% Plot the contour of posterior 
X = 40*[-1, 1];
Y = 40*[-1, 1];
X = linspace(X(1), X(2), 101);
Y = linspace(Y(1), Y(2), 101);
[X,Y] = meshgrid(X,Y);

Z = mvnpdf([X(:), Y(:)], mean(DecodeRes(1:2,:),2)', cov(DecodeRes(1:2,:)'));
contourf(X(1,:), Y(:,1), reshape(Z,size(X)), 'linestyle', 'none')

% Define the colormap of the same color series
cMap = getColorMapPosNegDat([0, max(Z)], 64);
colormap(cMap);
axis xy

% Plot the Gibbs sampling steps
nSteps = 5;
plot([DecodeRes(1,1:nSteps); DecodeRes(1,1:nSteps)], [DecodeRes(2,1:nSteps); DecodeRes(2,2:nSteps+1)], 'y')
plot([DecodeRes(1,1:nSteps); DecodeRes(1,2:nSteps+1)], [DecodeRes(2,2:nSteps+1); DecodeRes(2,2:nSteps+1)], 'y')
plot(DecodeRes(1,2:nSteps+1), DecodeRes(2,2:nSteps+1), 'oy', 'markersize', 6)
