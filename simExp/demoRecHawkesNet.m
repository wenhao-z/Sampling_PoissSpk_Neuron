% Demo of a linear Hawkes process where the E neurons are organized on a
% ring

% Wen-Hao Zhang, July 2, 2019
% University of Pittsburgh

%% Parameters of the model
parsHawkesNet;

% parsMdl.Ne = 180 * 40;
% parsMdl.tLen = 2e3*1e3; % unit: ms 
parsMdl.tLen = 100*1e3; % unit: ms 
parsMdl.bSample_ufwd = 1;
parsMdl.jxe = 4e-3; %2e-2;
parsMdl.ratiojie = 5;
parsMdl.FanoFactorIntVar = 1;

parsMdl.tTrial = 0.2*1e3; % length of a trial. Unit: ms

% Compute the dependent parameters
parsMdl = getDependentPars_HawkesNet(parsMdl);

%% Simulation

% Generate a sample of feedforward spiking input
ratefwd = makeRateFwd(parsMdl.Posi, parsMdl);
ratefwd = [ratefwd; parsMdl.ji0 * sum(ratefwd)/parsMdl.Ni*ones(parsMdl.Ni,1)];
% ratefwd = [ratefwd; zeros(parsMdl.Ni,1)];

tic
outSet = simHawkesNetDemo(ratefwd, parsMdl);
toc
%%
% Spike count in of each trial
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

% Calculate the correlation with distance on the ring
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
% figure(2); clf;
figure
subplot(2,3,1)
plot(parsMdl.PrefStim, NetStat.ratePop)
hold on;
plot(parsMdl.PrefStim, NetStat.ratePop + sqrt(diag(CovRate)));
plot(parsMdl.PrefStim, NetStat.ratePop - sqrt(diag(CovRate)));
set(gca, 'xlim', parsMdl.PrefStim(end)*[-1,1], 'xtick', -180:90:180)
ylabel('Firing rate (Hz)')
axis square

title(['T=' num2str(parsMdl.tLen/1e3), 's'])

subplot(2,3,2)
plot(1:parsMdl.Ni, mean(nSpkI,2)/parsMdl.tTrial*1e3)
hold on;
plot(1:parsMdl.Ni, (mean(nSpkI,2) + std(nSpkI, 0,2))/parsMdl.tTrial*1e3);
plot(1:parsMdl.Ni, (mean(nSpkI,2) - std(nSpkI, 0,2))/parsMdl.tTrial*1e3);
set(gca, 'xlim', [1, parsMdl.Ni], 'xtick', [1, parsMdl.Ni], ...
    'ylim', [0, 50])
ylabel('Firing rate (Hz)')
axis square

subplot(2,3,3)
plot(parsMdl.PrefStim, (diag(CovRate)./NetStat.ratePop))
set(gca, 'xlim', parsMdl.PrefStim(end)*[-1,1], 'xtick', -180:90:180)
ylabel('Fano factor')
ylim([1, 1.5])
axis square

subplot(2,3,4)
plot(parsMdl.PrefStim, diag(CovRate));
ylabel('Cov. of rate')
set(gca, 'xlim', parsMdl.PrefStim(end)*[-1,1], 'xtick', -180:90:180)
axis square

subplot(2,3,5)
imagesc(parsMdl.PrefStim, parsMdl.PrefStim, CovRate - diag(diag(CovRate)));
% imagesc(CorrRate - diag(diag(CorrRate)));
set(gca, 'xlim', parsMdl.PrefStim(end)*[-1,1], 'ylim', parsMdl.PrefStim(end)*[-1,1], ...
    'xtick', -180:90:180, 'ytick', -180:90:180)
axis xy
title('Cov. of rate')
axis square

subplot(2,3,6)
yyaxis left
plot(diffPrefStim(2:end), CovRate_diffPrefStim(2:end));
ylabel('Cov. of rate')
yyaxis right
plot(diffPrefStim(2:end), CorrRate_diffPrefStim(2:end));
ylabel('Corr. of rate')
set(gca, 'xlim', diffPrefStim([1, end]), 'xtick', 0:90:180)
axis square


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
% X = meanPosterior(1) + 3*covPosterior(1,1)*[-1, 1];
% Y = meanPosterior(2) + 3*covPosterior(2,2)*[-1, 1];
X = 40*[-1, 1];
Y = 40*[-1, 1];
X = linspace(X(1), X(2), 101);
Y = linspace(Y(1), Y(2), 101);
[X,Y] = meshgrid(X,Y);

Z = mvnpdf([X(:), Y(:)], mean(DecodeRes(1:2,:),2)', cov(DecodeRes(1:2,:)'));
contourf(X(1,:), Y(:,1), reshape(Z,size(X)), 'linestyle', 'none')
% imagesc(X(1,:), Y(:,1), reshape(Z,size(X)))

% Define the colormap of the same color series
cMap = getColorMapPosNegDat([0, max(Z)], 64);
% cMap = flipud(hot(64));
colormap(cMap);
axis xy

% Plot the Gibbs sampling steps
nSteps = 5;
plot([DecodeRes(1,1:nSteps); DecodeRes(1,1:nSteps)], [DecodeRes(2,1:nSteps); DecodeRes(2,2:nSteps+1)], 'y')
plot([DecodeRes(1,1:nSteps); DecodeRes(1,2:nSteps+1)], [DecodeRes(2,2:nSteps+1); DecodeRes(2,2:nSteps+1)], 'y')
plot(DecodeRes(1,2:nSteps+1), DecodeRes(2,2:nSteps+1), 'oy', 'markersize', 6)
