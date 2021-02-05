function Kerl = gaussTuneKerl(posi, TunWidth, parsMdl, bNormalize)
% Kernel of Gaussian tuning
% Wen-Hao Zhang, June 24, 2019

% Circular boundary
diff = exp(1i * (parsMdl.PrefStim - posi)/parsMdl.width*pi);
diff = angle(diff)*parsMdl.width/pi;

Kerl = exp(-diff.^2/ (2*TunWidth^2));

if bNormalize
    % Normalize into a pdf
    Kerl = Kerl/(sqrt(2*pi)*parsMdl.TunWidth);
end
