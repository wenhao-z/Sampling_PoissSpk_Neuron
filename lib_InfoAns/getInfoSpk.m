function InfoAnsRes = getInfoSpk(tSpkArray, NetPars, dimPar, bootstrap)
% Estimate the Fisher inforamtion of external and internal stimulus
% contained in spiking activities
% Wen-Hao Zhang, Oct 25, 2018
% wenhao.zhang@pitt.edu
% @University of Pittsburgh

if ~exist('minRateAns', 'var')
    % The minimal firing rate to include a cell into analylsis
    minRateAns = 0; % unit: hz
end

% The dim of Posi in parameter grid
namePar = {dimPar.namePar};
IdxPosiDim = find(cellfun(@(x) strcmp(x, 'Posi'), namePar));


%% Get spike count every trial (segment)
IdxStat = cellfun(@(x) find(x(1,:) > NetPars.tStat/NetPars.dt, 1, 'first'), ...
    tSpkArray);
IdxStat = num2cell(IdxStat);
tSpk_stat = cellfun(@(x, Idx) x(:, Idx:end), tSpkArray, IdxStat, 'uniformout', 0);

% Spike count of neurons in each trial and each stimulus condition
nSpk = cellfun(@(tSpk) accumarray(tSpk(2,:)', ones(1, size(tSpk, 2)), [NetPars.N, 1]), ...
    tSpk_stat, 'uniformout', 0);
nSpk = cell2mat(shiftdim(nSpk, -1));
nSpk = squeeze(nSpk); % [Neuron, trials, parGrid]
clear IdxStat tSpk_stat

% Permute the dim of nSpk
orderDim = [1:2, IdxPosiDim+2, setdiff(1:ndims(nSpk), [1:2, IdxPosiDim+2])];
nSpk = permute(nSpk, orderDim); % array: [N, trial, Posi, other parameters]

sznSpk = size(nSpk);
nSpk = reshape(nSpk, [sznSpk(1:3), prod(sznSpk(4:end))]);
nSpk = mat2cell(nSpk, sznSpk(1), sznSpk(2), sznSpk(3), ones(1, prod(sznSpk(4:end))) );
nSpk = reshape(nSpk, sznSpk(4:end)); % a cell array with dim as parameter grid without Posi
% Each cell contains a 3D array [N, nTrials, Posi]

%% Estimate the Fisher information of external stimulus by using bias correlation
% Ref: Kanitscheider, Plos Comp. Biol. 2015

[Info_sext, rateAvg, corrAvg] = cellfun(@(n) fisherInfo_biasCorrect(n, NetPars, minRateAns), nSpk);

if bootstrap
    % Bootstrap to estimate the std of information of external stimulus
    nBoot = 50;
    stdInfo_sext = cellfun(@(n) bootstrp_fisherInfo(n, NetPars, minRateAns, nBoot), nSpk);
end
%% Fisher information of internal stimulus (stimulus estimate)
% Gaussian tuning + independent Poisson spikes
Info_sint = cellfun(@(nAvg) sum(mean(mean(nAvg,2),3), 1)/NetPars.TunWidth^2, nSpk);
% stdInfo_sint = cellfun(@(nAvg) sum(std(mean(nAvg,3),0, 2), 1)/NetPars.TunWidth^2, nSpk);


%% Fold results into a output struct
InfoAnsRes.Info_sext = Info_sext;
InfoAnsRes.Info_sint = Info_sint;

InfoAnsRes.rateAvg = rateAvg;
InfoAnsRes.corrAvg = corrAvg;

if bootstrap
    InfoAnsRes.stdInfo_sext = stdInfo_sext;
end
end

