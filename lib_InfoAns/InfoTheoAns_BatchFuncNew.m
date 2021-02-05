function [InfoAnsRes, NetStat] = InfoTheoAns_BatchFuncNew(nSpk, popVec, parsMdl, varargin)
% Perform the information-theoretic analysis on network's responses
% Wen-Hao Zhang, July 9, 2019
% University of Pittsburgh

% nSpk: a cell. 1st dim: stim. position
% nSpk(1,:) the nSpk under different network parameters given Posi(1)
% nSpk(2,:) the nSpk under different network parameters given Posi(2)

% Optional parameters for inforamtion analysis
% minRateAns: the minimal firing rate of a neuron to be included into the
%             Fisher information analysis
% bBootStrap: using bootstrap to estimate the std of Fisher information
% nBootStrap: the number of bootstrap


% Get the possible parameters from varargin
if mod(size(varargin,2) , 2) == 1 % odd number input
    error('The varargin input number is wrong!')
else
    for iter = 1: round(size(varargin,2)/2)
        eval([varargin{2*iter-1} '= varargin{2*iter};']);
    end
end
clear iter

% Default parameters
if ~exist('minRateAns', 'var')
    % The minimal firing rate to include a cell into analysis
    minRateAns = 0; % unit: hz
end
if ~exist('bBootStrap', 'var')
    bBootStrap = 0;
    nBootStrap = 0;
end
if ~bBootStrap
    nBootStrap = 0;
end

%% Statistics of decoder's output (the position on the ring manifold)
NetStat = cellfun(@(nspk, popvec) getNetMdlStat(nspk, popvec, parsMdl), nSpk, popVec);
% Conver the struct array to a single struct with each field an array??

%% Classical Fisher information of external stimulus

% ------------------------------------------------------------
% Numerical estimate of Fisher information with bias correction
[InfoAnsRes.FisherInfo_sext, InfoAnsRes.stdFisherInfo_sext] ...
    = cellfun(@(n1,n2) fisherInfo_biasCorrect(n1, n2, parsMdl.dPosi*[-1,1]/2, parsMdl, ...
        'minRateAns', minRateAns, 'bBootStrap', bBootStrap, 'nBootStrap', nBootStrap), ...
        nSpk(1,:), nSpk(2,:));
    
% if bBootStrap
%     [~, InfoAnsRes.stdFisherInfo_sext] = cellfun(@(n1,n2) fisherInfo_biasCorrect(n1, n2, parsMdl.dPosi*[-1,1]/2, parsMdl, ...
%         'minRateAns', minRateAns, 'bBootStrap', bBootStrap, 'nBootStrap', nBootStrap), ...
%         nSpk(1,:), nSpk(2,:));    
% end

% ------------------------------------------------------------
% Fisher information of shuffled data 
% Motivation: demo the info. of internal stimulus can be estimated so simply!! 

% Shuffle the original neuronal responses across trials
for iter = 1: parsMdl.Ne
   nSpk{1}(iter,:) = nSpk{1}(iter,randperm(parsMdl.nTrials));
   nSpk{2}(iter,:) = nSpk{2}(iter,randperm(parsMdl.nTrials));   
end
[InfoAnsRes.FisherInfo_shuffled, InfoAnsRes.stdFisherInfo_shuffled]...
    = cellfun(@(n1,n2) fisherInfo_biasCorrect(n1, n2, parsMdl.dPosi*[-1,1]/2, parsMdl, ...
    'minRateAns', minRateAns, 'bBootStrap', bBootStrap, 'nBootStrap', nBootStrap), ...
    nSpk(1,:), nSpk(2,:));

% if bBootStrap
%     [~, InfoAnsRes.stdFisherInfo_shuffled] = cellfun(@(n1,n2) fisherInfo_biasCorrect(n1, n2, parsMdl.dPosi*[-1,1]/2, parsMdl, ...
%         'minRateAns', minRateAns, 'bBootStrap', bBootStrap, 'nBootStrap', nBootStrap), ...
%         nSpk(1,:), nSpk(2,:));
% end

% ------------------------------------------------------------
% Theoretical prediction of linear Fisher information
[InfoAnsRes.FisherInfo_sext_theory, InfoAnsRes.FisherInfo_sint_theory] = arrayfun(@(S) ...
    fisherInfo_Theory(S.rateHeight, S.tuneWidth, S.rateOffset, S.covSample(3,3), parsMdl), ...
    NetStat(1,:));
% A simple theoretical prediction of linear Fisher information
[InfoAnsRes.FisherInfo_sext_theorySimp, InfoAnsRes.FisherInfo_sint_theorySimp] = arrayfun(@(S) ...
    fisherInfo_TheorySimple(S.rateAvg, parsMdl.Ne, S.tuneWidth, S.covSample(3,3)), ...
    NetStat(1,:));

%% The cross entropy, mutual information

% Note: only use the network responses under stimulus position 1 to
% estimate the mutual information! 
[InfoAnsRes.XEnt, InfoAnsRes.MutualInfo, InfoAnsRes.MutualInfo_UpBound] = ...
    arrayfun(@(S) getMutualInfo(parsMdl.meanPosterior, parsMdl.covPosterior, ...
    S.meanSample(1:2), S.covPostSample, 2*parsMdl.width), NetStat(1,:));

end
