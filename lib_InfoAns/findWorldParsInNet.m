function UrecWorld = findWorldParsInNet(parsMdl, meanSample, covSample)
% Given the network parameter and responses find the world parameter which
% the network could maximize the mutual information.

% CAUTION: this code is used for results after scanning network parameters
% Wen-Hao Zhang, July 22, 2019
% wenhao.zhang@pitt.edu
% University of Pittsburgh

gridUrecWorld = 0:0.1:60;
meanPosterior = parsMdl.Posi - parsMdl.dPosi/2;
meanPosterior = repmat(meanPosterior, 2, 1);

% Assign a grid of UrecWorld
nameFields = fieldnames(parsMdl);
valueFields = struct2cell(parsMdl);
valueFields = repmat(valueFields, [1, length(gridUrecWorld)]);

IdxUrec = cellfun(@(x) strcmp(x, 'UrecWorld'), nameFields);
valueFields(IdxUrec,:) = num2cell(gridUrecWorld);

IdxPosi = cellfun(@(x) strcmp(x, 'Posi'), nameFields);
valueFields(IdxPosi,:) = repmat({meanPosterior}, 1, length(gridUrecWorld));

parsMdl = cell2struct(valueFields, nameFields);


% Given each world parameter, calculate the posterior parameters
[meanPosterior, covPosterior] = arrayfun(@(S) getPosteriorHierMdl(S), parsMdl, 'uniformout', 0);

[~, MutualInfo] = cellfun(@(mean, cov) getMutualInfo(mean, cov, meanSample(1:2), covSample(1:2,1:2), 2*parsMdl(1).width), ...
    meanPosterior, covPosterior);

[~, IdxMax] = max(MutualInfo);

UrecWorld = gridUrecWorld(IdxMax);

end