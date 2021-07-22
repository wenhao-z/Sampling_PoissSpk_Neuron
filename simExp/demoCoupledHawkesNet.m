% Demo of a linear Hawkes process where the E neurons are organized on a
% ring

% Wen-Hao Zhang, July 2, 2019
% University of Pittsburgh

%% Parameters of the model
parsCoupledHawkesNet;

parsMdl.dt = 0.1;
parsMdl.tBin = 20; % length of a trial. Unit: ms

parsMdl.bSample_ufwd = 0;
parsMdl.jxe = 1e-2;
parsMdl.ratiojie = 5;
parsMdl.ratiojrprc = 5;
parsMdl.bCutRecConns = 1; % Cut off all recurrent connections to simplify theoretical analysis
parsMdl.bShareInhPool = 0; % Whether two networks share the same inhibition pool

% Input parameters
parsMdl.Posi = 10*[-1; 1];
parsMdl.Ufwd = 30*[1;1];
parsMdl.tLen = 102*1e3; % unit: ms
parsMdl.tStat = 2*1e3;

parsMdl = getDependentPars_HawkesNet(parsMdl);

%% Simulation
NetStat = struct('tSample', [], ...
    'meanSample', [], ...
    'meanSamplePred', [], ...
    'covSample', [], ...
    'PreMat_LH', [], ...
    'ratePop', [], ...
    'rateHeight', []);

% Generate a sample of feedforward spiking input
ratefwd = makeRateFwd(parsMdl.Posi, parsMdl);
ratefwd = [ratefwd; parsMdl.ji0 * sum(ratefwd,1)./parsMdl.Ni.*ones(parsMdl.Ni,1)];

% Get the precision of the likelihood
preLH = sum(ratefwd(1:parsMdl.Ne,:),1)/ parsMdl.TunWidth^2 /2 * parsMdl.tBin /parsMdl.dt; % Precision of the likelihood
NetStat.PreMat_LH = diag(preLH);

tic
outSet = simCoupledHawkesNet(ratefwd(:), parsMdl);
toc

%%

% Spike count in of each trial
tEdge = [parsMdl.tStat+parsMdl.dt, parsMdl.tLen];
neuronEdge = [0.5: parsMdl.Ne + 0.5, parsMdl.Ncells+ (0.5: parsMdl.Ne + 0.5)];
nSpk = histcounts2(outSet.tSpk(1,:)', outSet.tSpk(2,:)', neuronEdge, tEdge);
nSpk(parsMdl.Ne+1) = [];
NetStat = getNetMdlStat(nSpk, outSet.popVec, parsMdl, NetStat);


% Numerically find the prior precision stored in the network by comparing
% posterior and likelihood
prePrior = arrayfun(@(x) findPriorPrecision(x.meanSample, x.covSample, parsMdl.Posi, x.PreMat_LH), ...
    NetStat);

%%

tLim = 5e2;
tSpk = outSet.tSpk;
tSpk(:, tSpk(2,:)>tLim) = [];

plotSpkRaster(tSpk, parsMdl, 'tBin', 0.01*1e3);
set(gca, 'xlim', [0, tLim])

%%
figure

figure
subplot(2,2,1)
plot(parsMdl.PrefStim, NetStat.ratePop)
set(gca, 'xlim', parsMdl.PrefStim(end)*[-1,1], 'xtick', -180:90:180)
ylabel('Firing rate (Hz)')


%%
figure; clf;
xyLim = 30;

hAxe = plotJointMarginalHist(NetStat.tSample(1,:), NetStat.tSample(2,:));
axes(hAxe(1))
xlabel('Stimulus feature s_1')
ylabel('Stimulus feature s_2')
% axis(hAxe(1), xyLim*[-1,1,-1,1])
axis(hAxe(1), [-30, 20, -20, 30])
set(hAxe(2), 'ylim', [0, 0.1])
set(hAxe(3), 'xlim', [0, 0.1])


% Plot the contour of posterior
X = NetStat.meanSample(1) + 3* sqrt(NetStat.covSample(1,1))*[-1, 1];
Y = NetStat.meanSample(2) + 3* sqrt(NetStat.covSample(2,2))*[-1, 1];
X = linspace(X(1), X(2), 201);
Y = linspace(Y(1), Y(2), 201);
[X,Y] = meshgrid(X,Y);

Z = mvnpdf([X(:), Y(:)], NetStat.meanSample', NetStat.covSample);
contourf(X(1,:), Y(:,1), reshape(Z,size(X)), 'linestyle', 'none', 'levelstep', 1e-4)
% imagesc(X(1,:), Y(:,1), reshape(Z,size(X)))

% Define the colormap of the same color series
cMap = getColorMapPosNegDat([0, max(Z)], 64);
% cMap = flipud(hot(64));
colormap(cMap);
axis xy

% Plot the Gibbs sampling steps
nSteps = 10;
tStep = 1e3+40;

plot([NetStat.tSample(1,tStep+(1:nSteps)); NetStat.tSample(1,tStep+(1:nSteps))], ...
    [NetStat.tSample(2,tStep+(1:nSteps)); NetStat.tSample(2,tStep+(2:nSteps+1))], 'y')
plot([NetStat.tSample(1,tStep+(1:nSteps)); NetStat.tSample(1,tStep+(2:nSteps+1))], ...
    [NetStat.tSample(2,tStep+(2:nSteps+1)); NetStat.tSample(2,tStep+(2:nSteps+1))], 'y')
plot(NetStat.tSample(1,tStep+(2:nSteps+1)), NetStat.tSample(2,tStep+(2:nSteps+1)), 'oy', 'markersize', 6)

% Plot some other auxiliary lines
plot(hAxe(1), parsMdl.Posi(1)*ones(1,2), xyLim*[-1,1], '--k')
plot(hAxe(1), xyLim*[-1,1], parsMdl.Posi(2)*ones(1,2), '--k')
plot(hAxe(1), xyLim*[-1,1], xyLim*[-1,1], '--k')
% clear X Y Z

%%
figure
cLevel = 2e-4;

X = NetStat.meanSample(1) + 3* sqrt(NetStat.covSample(1,1))*[-1, 1];
Y = NetStat.meanSample(2) + 3* sqrt(NetStat.covSample(2,2))*[-1, 1];
X = linspace(X(1), X(2), 201);
Y = linspace(Y(1), Y(2), 201);
[X,Y] = meshgrid(X,Y);
% Likelihood
hAxe(1) = subplot(1,3,1); hold on;
Z_LH = mvnpdf([X(:), Y(:)], parsMdl.Posi', inv(NetStat.PreMat_LH));
contourf(X(1,:), Y(:,1), reshape(Z_LH,size(X)), 'linestyle', 'none', 'levelstep', cLevel)
axis square

% Prior
hAxe(2) = subplot(1,3,2); hold on;
Z_Prior = X-Y;
Z_Prior(Z_Prior<-parsMdl.width) = Z_Prior(Z_Prior<-parsMdl.width) + 2*parsMdl.width;
Z_Prior(Z_Prior>parsMdl.width) = Z_Prior(Z_Prior>parsMdl.width) - 2*parsMdl.width;

Z_Prior = exp(-prePrior * Z_Prior.^2/2);
Z_Prior = Z_Prior ./ sum(Z_Prior(:)) / mean(diff(X(1,:))) / mean(diff(Y(:,1)));
contourf(X(1,:), Y(:,1), reshape(Z_Prior,size(X)), 'linestyle', 'none', 'levelstep', cLevel)
axis square

% Posterior
hAxe(3) = subplot(1,3,3); hold on;
Z = mvnpdf([X(:), Y(:)], NetStat.meanSample', NetStat.covSample);
contourf(X(1,:), Y(:,1), reshape(Z,size(X)), 'linestyle', 'none', 'levelstep', cLevel)
axis square

cMax = max([Z_LH(:); Z_Prior(:); Z(:)]);
cMap = getColorMapPosNegDat([0, cMax], 64);
colormap(cMap);

for iter = 1:3
    axes(hAxe(iter));
    plot(parsMdl.Posi(1)*ones(1,2), xyLim*[-1,1], '--k')
    plot(xyLim*[-1,1], parsMdl.Posi(2)*ones(1,2), '--k')
    plot(xyLim*[-1,1], xyLim*[-1,1], '--k')
    caxis(hAxe(iter), [0, cMax])
    
    box on
end

axes(hAxe(1)); title('Likelhood')
xlabel('Stimulus feature s_1')
ylabel('Stimulus feature s_2')
axes(hAxe(2)); title('Prior')
axes(hAxe(3)); title('Posterior')

linkaxes(hAxe, 'xy')
set(hAxe, 'xlim', [-25, 20], 'ylim', [-20, 25])