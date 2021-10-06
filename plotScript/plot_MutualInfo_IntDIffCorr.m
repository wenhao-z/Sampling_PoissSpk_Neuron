% Plot the results of Mutual information and total differential correlation

% Wen-Hao Zhang
% Aug. 16, 2021

% Load data
setWorkPath;

addpath(fullfile(Path_RootDir, 'lib_InfoAns'));

datPath = fullfile(Path_RootDir, 'Data');
Folder = 'HawkesNet';
% fileName = 'SingleRecNet_210813_0009.mat';
fileName = 'SingleRecNet_211005_0358.mat';


load(fullfile(datPath, Folder, fileName));


%% Compute the statistics of samples and posterior distribution

preSample = arrayfun(@(S) inv(S.covSample(1:2,1:2)), NetStat, 'uniformout', 0);
preSample = cell2mat(shiftdim(preSample,-2));
preSample = squeeze(preSample);

preSamplePred = reshape([NetStat.PreMat_LH], [2,2, size(parGrid)]);

% Subjective prior
prePrior = -squeeze(preSample(1,2,:));
% prePrior = squeeze(preSample(2,2,:));
prePrior = reshape(prePrior, size(NetStat));
preSamplePred = preSamplePred + shiftdim(prePrior,-2) .* repmat([1,-1;-1,1], [1,1, size(prePrior)]);

%% Compute the mutual information

% A fixed prior in the world
parsMdl.UrecWorld = 10;
% prePrior = parsMdl.UrecWorld * NetStat(end).PreMat_LH(1) / parGrid(end).Ufwd;

% Calculate covariance of samples by subtracting the cov. of feedfoward cue
for iter = 1: numel(NetStat)
    if parsMdl.bSample_ufwd
        covSample = NetStat(iter).covSample;
        NetStat(iter).covSample(1:2,1:2) = covSample(1:2,1:2) - covSample(1,4)^2/ covSample(4,4);
    end
end
% clear covSample

% Compute the covariance of posterior
for iter = 1: numel(parGrid)
    parGrid(iter).UrecWorld = parsMdl.UrecWorld; 
end
[~, covPosterior] = arrayfun(@(netpars) getPosteriorHierMdl(netpars), parGrid, 'uniformout', 0);


% Mutual information
[~, MutualInfo, MutualInfo_UpBound] = arrayfun(@(S, covPost) ...
    getMutualInfo(zeros(2,1), covPost{1}, ...
    S.meanSample(1:2), S.covSample(1:2,1:2), 2*parsMdl.width), ...
    NetStat, covPosterior);

%%
figure
IdxJxe = 5;
IdxUfwd = 6;

covSample = reshape([NetStat.covSample], [4,4, size(NetStat)]);

subplot(1,2,1)
yyaxis left
plot(parsMdl.Ufwd(2:end-2), MutualInfo(IdxJxe, 2:end-2, 2)./ parsMdl.tBin*1e3)
ylabel('Mutual info. (bit/sec.)')

yyaxis right
plot(parsMdl.Ufwd(2:end-2), squeeze(covSample(3,3,IdxJxe, 2:end-2,2)))
ylabel('Total diff. corr.')
axis square
xlabel('Feedforward rate (Hz)')
title(['jxe = ' num2str(parsMdl.jxe(IdxJxe))])

subplot(1,2,2)
yyaxis left
plot(parsMdl.jxe(1:end/2), MutualInfo(1:end/2,IdxUfwd,1)./ parsMdl.tBin*1e3)
% hold on;
% plot(parsMdl.jxe(1:end/2), MutualInfo(1:end/2,IdxUfwd,2)./ parsMdl.tBin*1e3)
ylabel('Mutual info. (bit/sec.)')
set(gca, 'yticklabel', 110:20:170)

yyaxis right
plot(parsMdl.jxe(1:end/2), squeeze(covSample(3,3,1:end/2,IdxUfwd,1))+ squeeze(covSample(4,4,1:end/2,IdxUfwd,2)))
ylabel('Total diff. corr.')
ylim([76.5, 78])
set(gca, 'ytick', 76.5:0.5:78)
axis square
xlim([0, 5e-3])
xlabel('Rec weight w_E')
title(['U_{fwd} = ' num2str(parsMdl.Ufwd(IdxUfwd))])

%%
figure
plot(parsMdl.jxe(1:end/2), squeeze(covSample(3,3,1:end/2,IdxUfwd,1)))
xlabel('Rec weight w_E')
ylabel('Internal diff. corr.')


% Predict the int. diff. corr. through subtracting the var. of sample with
% the var. of conditional distribution.

% Fit the tunning width
% tuneParams = arrayfun(@(S) lsFitTuneFunc(S.ratePop, parsMdl), NetStat, 'uniformout', 0);
% tuneWidth = cellfun(@(S) S(3), tuneParams);
% varCondDistri = tuneWidth.^2 ./ arrayfun(@(S) sum(S.ratePop), NetStat);
% varCondDistri = varCondDistri / parsMdl.tBin * 1e3;
% 
% hold on
% plot(parsMdl.jxe(1:end/2), ...
% squeeze(covSample(1,1,1:end/2,IdxUfwd,1)) - varCondDistri(1:end/2,IdxUfwd,1));
