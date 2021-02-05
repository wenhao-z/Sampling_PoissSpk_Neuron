function tSpk = IndPoissSpkGen(rate, parsMdl)
% Independent Poisson spike generator to simulate a population of neurons
% Wen-Hao Zhang, Aug. 6, 2019
% wenhao.zhang@pitt.edu

% The unit of inputs
% rate: unit of Hz
% dt:   unit of ms

% Output
% tSpk(1,:): index of spiking neurons
% tSpk(2,:): spike timing

% Load the random generator 
rng(parsMdl.rngNetSpk);

IdxSpk = find(rate*parsMdl.dt/1e3 > ...
    rand(parsMdl.Ne, parsMdl.tLen/parsMdl.dt));

[tSpk(1,:), tSpk(2,:)] = ind2sub([parsMdl.Ne, parsMdl.tLen/parsMdl.dt], IdxSpk);