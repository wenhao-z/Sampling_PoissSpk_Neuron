function [Param, err] = lsFitTuneFunc(rate, parsMdl)
% Least square fit of the tuning curves by 
% Height * exp(-(\theta- posi)^2/ (2*Width^2)) + Biase
% Param = [Height, posi, Width, Bias];

% Wen-Hao Zhang, Aug. 6, 2019
% wenhao.zhang@pitt.edu

% Set the initial value of parameters
posi = parsMdl.PrefStim'*rate/ sum(parsMdl.PrefStim);
Width = sqrt(sum((parsMdl.PrefStim - posi).^2 .* rate) ./sum(rate,1));
Height = sum(rate, 1) ./ (sqrt(2*pi)*parsMdl.rho*Width);
Bias = min(rate);
Param0 = [Height, posi, Width, Bias];


options = optimset('TolX', 1e-7, 'display', 'off');
[Param, err] = fminsearch(@(x) tuneFunc(x), Param0, options);

    function err = tuneFunc(Param)
        % Param = [Height, Posi, Width, Bias]
        y = Param(1) * gaussTuneKerl(Param(2), Param(3), parsMdl, 0) + Param(4);
        
        err = sum((y-rate).^2);
    end
end