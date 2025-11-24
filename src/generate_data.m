%--------------------------------------------------------------------------
% GENERATE_DATA
% 
% Generate a synthetic data tensor.
%
% input:  XI, a vector contains the dimensionality of the data tensor X
%         YI, a vector contains the dimensionality of the data tensor Y
%         R, rank of data tensor
%         SNR, Noise level in dB, inf for noiseless tensor
%         TYPE, stream type (single-tensor or matrix-tensor)
% ouputs: X, the output data tensor X
%         Y, the output data tensor Y
%--------------------------------------------------------------------------

function [ X, Y ] = generate_data( XI, YI, R, SNR, TYPE )

if TYPE == "ST"
    if XI(1)~=YI(1)
        error('When TYPE is "ST", the first modes of X and Y must have the same size.');
    end
    XA = arrayfun(@(x) rand(x, R), XI, 'uni', 0);
    YA = [XA(1), arrayfun(@(x) rand(x, R), YI(2:end), 'uni', 0);]; % Share the first axis
elseif TYPE == "MT"
    if XI(end)~=YI(1)
        error('When TYPE is "MT", the last mode of X and the first mode of Y must have the same size.');
    end
    XA = arrayfun(@(x) rand(x, R), XI, 'uni', 0);
    YA = [XA(end), arrayfun(@(x) rand(x, R), YI(2:end), 'uni', 0);]; % Share the last axis
else
    error('Unknown TYPE error. One of "ST" or "MT" is allowed.');
end

% Construct tensors
X = ktensor(XA(:));
X = full(X);
Y = ktensor(YA(:));
Y = full(Y);

% Add noise
X_noise = randn(XI);
X_sigma = (10^(-SNR/20)) * (norm(X)) / norm(tensor(X_noise));
X       = double(X) + X_sigma * X_noise;
Y_noise = randn(YI);
Y_sigma = (10^(-SNR/20)) * (norm(Y)) / norm(tensor(Y_noise));
Y       = double(Y) + Y_sigma * Y_noise;

end