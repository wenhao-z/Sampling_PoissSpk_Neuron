function nSpk = tSpk2nSpk(tSpk, tSeg, parsMdl)
% Convert the spike timing to the spike count
% tSpk: cell(size(parGrid)), 
%            each cell element of tSpkArray is [2, spike timing]  array
% nSpk:      cell(size of parGrid except Posi), 
%            each cell element is a 3D array [Ne, nTrials, Posi]
% The two input positions are used to estimate of Fisher information of
% external stimulus

% Wen-Hao Zhang, July 8, 2019
% University of Pittsburgh


% Get the spike count of each trial
tEdge = [0: tSeg : parsMdl.tLen, parsMdl.tLen + tSeg];
neuronEdge = 0.5: parsMdl.Ne + 0.5;
nSpk = histcounts2(tSpk(1,:)', tSpk(2,:)', neuronEdge, tEdge);
nSpk = nSpk(:,1:end-1);