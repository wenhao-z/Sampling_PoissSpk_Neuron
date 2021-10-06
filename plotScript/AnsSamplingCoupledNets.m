% Analyze the result of sampling-based inference in coupled neural networks

% Wen-Hao Zhang, June 22, 2021
% wenhaoz@uchicago.edu
% University of Chicago

% Load simulation data

% Load data
setWorkPath;

datPath = fullfile(Path_RootDir, 'Data', 'HawkesNet');
fileName = 'CoupledNet_210624_0135.mat';
% fileName = 'CoupledNet_210623_0526.mat';

load(fullfile(datPath, fileName));

%%
% Numerically find the prior precision stored in the network by comparing
% posterior and likelihood
if ~exist('prePrior', 'var')
    prePrior = arrayfun(@(x) findPriorPrecision(x.meanSample, x.covSample, parsMdl.Posi, x.PreMat_LH), ...
        NetStat);
end
preSamplePred = reshape([NetStat.PreMat_LH], [parsMdl.numNets, parsMdl.numNets, size(NetStat)]);
preSamplePred = preSamplePred + ...
    shiftdim(prePrior,-2) .* repmat([1,-1;-1,1], [1,1, size(NetStat)]);

%% Figure
figure

cSpec = lines(1);

subplot(1,2,1); hold on
xyLim = 10;
plot(reshape([NetStat.meanSample], 1, []), reshape([NetStat.meanSamplePred], 1, []), '.', ...
    'color', cSpec, 'markersize', 8)
% plot(parsMdl.Posi, parsMdl.Posi, '--k')
plot(xyLim*[-1,1], xyLim*[-1,1], '--k')
xlim(xyLim*[-1,1])
ylim(xyLim*[-1,1])
axis square
xlabel('Sampling mean')
ylabel('Posterior mean')

subplot(1,2,2)
hold on
% plot(preSample(:), preSamplePred(:), '.')

for iter = 1: numel(NetStat)
    plot(diag(preSample(:,:, iter)), diag(preSamplePred(:,:, iter)), '.', ...
        'color', cSpec, 'markersize', 8);
    %     plot(-preSample(1,2, iter), preSamplePred(1,2, iter), '.');
end
plot([0, 0.05], [0, 0.05], '--k')
axis square
xlabel('Sampling precision')
ylabel('Posterior precision')