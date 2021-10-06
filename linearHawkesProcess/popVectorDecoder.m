function [Posi, popVec, meanPosi, covPosi] = popVectorDecoder(Input, parsMdl)
% Decode the bump position by using population vector
% That is, the center of mass in the Gaussian case

% Input could be
% 1) Neuron activity, then the length of 1st dim of Input should be the
%    same as parsMdl.Ne
% 2) The population vector, a complex number.

% Wen-Hao Zhang, July 9, 2019

if size(Input,1) == parsMdl.Ne
    popVec = popVectorRep(Input, parsMdl);
elseif ~isreal(Input) || (sum(abs(Input(:))) ==0)
    popVec = Input;
else
    error('Input is wrong!')
end

% ---------------------------------------------------------------
% Get the equilibrium time point
if isfield(parsMdl, 'tStat')
    popVec = popVec(:, parsMdl.tStat/parsMdl.dt+1 : end);
end

% ---------------------------------------------------------------
% Merge the popVec in the same time window
if isfield(parsMdl, 'tBin')
    nBin = parsMdl.tBin/parsMdl.dt;
    
    subsAvg = [repmat(1:size(popVec,1), 1, size(popVec,2)); ...
        kron(1: size(popVec,2)/nBin, ones(1, round(size(popVec,1)*nBin)))];
    popVec = accumarray(subsAvg', popVec(:), []);
end

% Convert complex representation into real position
Posi = angle(popVec) * parsMdl.width / pi;
% For zero activity, randomly draw a position from feature space
Idx = (popVec == 0);
% Posi(Idx) = nan;
Posi(Idx) = (2*rand(1, sum(Idx(:)))-1)*parsMdl.width;

if nargout > 2
    meanPosi = mean(popVec, 2, 'omitnan'); % complex number
    % Angular
    meanPosi = angle(meanPosi)* parsMdl.width/pi; % angular value
end

if nargout > 3
    % Variance
    devBumpPos = bsxfun(@minus, Posi, meanPosi);
    devBumpPos(devBumpPos>parsMdl.width) = devBumpPos(devBumpPos>parsMdl.width) - 2*parsMdl.width;
    devBumpPos(devBumpPos<parsMdl.width) = devBumpPos(devBumpPos<parsMdl.width) + 2*parsMdl.width;
    
    covPosi = cov(devBumpPos', 'omitrows');
end
