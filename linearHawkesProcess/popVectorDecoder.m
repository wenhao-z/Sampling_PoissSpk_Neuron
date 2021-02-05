function [Posi, popVec] = popVectorDecoder(Input, parsMdl)
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

Posi = angle(popVec) * parsMdl.width / pi;
% For zero activity, randomly draw a position from feature space
Idx = (popVec == 0);
Posi(Idx) = (2*rand(1, sum(Idx(:)))-1)*parsMdl.width;
