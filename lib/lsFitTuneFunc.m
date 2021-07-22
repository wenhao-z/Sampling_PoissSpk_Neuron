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
Param0 = [Height; posi; Width; Bias];

Param = zeros(size(Param0));
options = optimset('TolX', 1e-7, 'display', 'off');
for iter = 1: size(rate, 2)
    [Param(:,iter), err] = fminsearch(@(x) tuneFunc(x,rate(:,iter)), Param0(:,iter), options);
end

    function err = tuneFunc(Param, y0)
        % Param = [Height, Posi, Width, Bias]
        y = Param(1) * gaussTuneKerl(Param(2), Param(3), parsMdl, 0) + Param(4);
        
        err = sum((y-y0).^2);
    end
end