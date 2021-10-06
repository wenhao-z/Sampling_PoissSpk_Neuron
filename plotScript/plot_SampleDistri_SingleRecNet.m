% Plot the results of Mutual information and total differential correlation

% Wen-Hao Zhang
% Aug. 16, 2021

% Load data
setWorkPath;

addpath(fullfile(Path_RootDir, 'lib_InfoAns'));

datPath = fullfile(Path_RootDir, 'Data');
Folder = 'HawkesNet';
% fileName = 'SingleRecNet_210706_1934.mat';
% fileName = 'SingleRecNet_RandPars.mat';
% fileName = 'SingleRecNet_210807_1704_DiffSeed.mat';
% fileName = 'SingleRecNet_RandPars_210818_2334.mat';
fileName = 'SingleRecNet_RandPars_210819_1259.mat';
% fileName = 'SingleRecNet_RandPars_OldRecVar_210824.mat';

load(fullfile(datPath, Folder, fileName));


%% Compute the sampling distribution and the prediction of posterior
preSample = arrayfun(@(S) inv(S.covSample(1:2,1:2)), NetStat, 'uniformout', 0);
preSample = cell2mat(shiftdim(preSample,-2));
preSample = squeeze(preSample);

preSamplePred = reshape([NetStat.PreMat_LH], [2,2, size(parGrid)]);

% Numerically find the prior precision stored in the network by comparing
% posterior and likelihood
prePrior = arrayfun(@(x) ...
    findPriorPrecision(x.meanSample, x.covSample, parsMdl.Posi * ones(2,1), x.PreMat_LH), ...
    NetStat);

% prePrior = -squeeze(preSample(1,2,:));
prePrior = reshape(prePrior, size(NetStat));
preSamplePred = preSamplePred + shiftdim(prePrior,-2) .* repmat([1,-1;-1,1], 1,1, length(prePrior));

%% 
figure
cSpec = lines(2);
plot(squeeze(preSample(1,1,:)), squeeze(preSamplePred(1,1,:)), '.', 'markersize', 10, 'color', cSpec(1,:))
hold on
plot(squeeze(preSample(2,2,:)), squeeze(preSamplePred(2,2,:)), '.', 'markersize', 10, 'color', cSpec(2,:))

plot([0, max(preSample(:))], [0, max(preSample(:))], '--k')
legend('Stimulus sample', 'Context sample')
axis square; box off

xlabel('Precision of samples')
ylabel('Posterior precision')
