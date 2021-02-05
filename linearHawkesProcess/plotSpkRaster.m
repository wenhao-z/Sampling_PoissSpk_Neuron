function hAxe = plotSpkRaster(tSpk, parsNet, varargin)
% Generate a raster plot to demonstarte spike times for a E-I network model

% Wen-Hao Zhang
% wenhao.zhang@pitt.edu
% University of Pittsburgh
% Feb. 5, 2019


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Get the possible parameters from varargin
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Use this to specify "tBin" to make a PSTH
if mod(size(varargin,2) , 2) == 1 % odd number input
    error('The varargin input number is wrong!')
else
    for iter = 1: round(size(varargin,2)/2)
        eval([varargin{2*iter-1} '= varargin{2*iter};']);
    end
end
clear iter

%%
tSpkE = tSpk(:, tSpk(1,:) <= parsNet.Ne);
tSpkI = tSpk(:, tSpk(1,:) >  parsNet.Ne);

figure;
hold on

% ------------------------------------------------------
% Rastergram of E neurons 
hAxe(1) = subplot(6,1, 1:4);
scatter(tSpkE(2,:), parsNet.PrefStim(tSpkE(1,:)), .1, 'b.')
ylim(parsNet.PrefStim([1,end]))
set(gca, 'ytick', parsNet.PrefStim([1,end/2,end]), 'yticklabel', parsNet.width*[-1,0,1])
set(gca, 'xtick', [])
ylabel('Neuron index')

% Rastergram of I neurons
hAxe(2) = subplot(6,1, 5);
scatter(tSpkI(2,:), tSpkI(1,:)-parsNet.Ne, .1, 'r.')
ylim([0, parsNet.Ni])
set(gca, 'xtick', [])

% ------------------------------------------------------
% Mean firing rate of E and I neurons across time
hAxe(3) = subplot(6,1,6);
hold on

if ~exist('tBin', 'var')
%     tBin = 5; % unit: ms
    tBin = parsNet.dt; % unit: ms
end
tEdge = [0: tBin : parsNet.tLen, parsNet.tLen + tBin/2];

neuronEdge = [1, parsNet.Ne + 0.5];
bSpkE = histcounts2(tSpk(1,:)', tSpk(2,:)', neuronEdge, tEdge);
bSpkE = bSpkE/ parsNet.Ne/ tBin * 1e3;

neuronEdge = [parsNet.Ne+1, parsNet.Ncells+0.5];
bSpkI = histcounts2(tSpk(1,:)', tSpk(2,:)', neuronEdge, tEdge);
bSpkI = bSpkI/ parsNet.Ni/ tBin * 1e3;

stairs(tEdge(1:end-1), bSpkE, 'b')
stairs(tEdge(1:end-1), bSpkI, 'r')

% Specify the size of time bin
axisRange = axis;

text(axisRange(2), axisRange(4), sprintf('tBin=%dms', tBin*1e3), ...
    'horizontalalignment', 'right', 'verticalalignment', 'top');

xlim([0, parsNet.tLen])
xlabel('Time (ms)')
ylabel('Rate (Hz)')
linkaxes(hAxe, 'x')

