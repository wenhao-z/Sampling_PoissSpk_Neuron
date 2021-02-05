function rateUfwd = makeRateFwd(Posi, parsMdl)
% Generate feedforward spiking input
% Wen-Hao Zhang, July 2, 2019
% University of Pittsburgh

% rateUfwd = parsMdl.Ufwd * gaussTuneKerl(Posi, parsMdl.TunWidth, parsMdl, 0);
rateUfwd = parsMdl.Ufwd * gaussTuneKerl(Posi, sqrt(2)*parsMdl.TunWidth, parsMdl, 0);
rateUfwd = rateUfwd * parsMdl.dt/1e3;

% if parsMdl.bSample_ufwd
%     ufwd = (rateUfwd * parsMdl.dt/1e3 > rand(parsMdl.Ne,1));
% else
%     ufwd = rateUfwd * parsMdl.dt/1e3;
% end