function tSpk = tSpkFormatConverter(tSpk, Ncells)
% Convert tSpk between two formats
% Format 1: [2, nSpks] array, 1st row is neuron index, and 2st row is spike timing
% Format 2: [Ncells, 1] cell. Each element stores the spike timing of corresponding neuron

% Wen-Hao Zhang
% Math Department, University of Pittsburgh
% wenhao.zhang@pitt.edu
% Feb 9, 2019


if iscell(tSpk)
    % Format 2 => 1
    tSpk = cell2mat(tSpk)';
else
    tSpk = accumarray(tSpk(1,:)', tSpk(2,:)', [Ncells, 1], @(x) {sort(x)});
end