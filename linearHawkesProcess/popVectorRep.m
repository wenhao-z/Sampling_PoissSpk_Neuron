function popVec = popVectorRep(r, parsMdl)
% Project the neuronal activity onto the unit circle

% r should be [N, nSamples] array;

% Wen-Hao Zhang, June 24, 2019

cirPos = exp(1i * parsMdl.PrefStim * pi/parsMdl.width);
popVec = squeeze(sum(cirPos .* r, 1));
