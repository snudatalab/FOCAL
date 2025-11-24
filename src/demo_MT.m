%--------------------------------------------------------------------------
% DEMO_MT
% 
% This is an implementation of "Fast and Accurate Online Coupled 
% Matrix-Tensor Factorization via Frequency Regularization"
% This code is designed for a matrix-tensor streaming setting.
% 
% To run the code, Tensor Toolbox is required.
% Brett W. Bader, Tamara G. Kolda and others. MATLAB Tensor Toolbox 
% Available online (https://www.tensortoolbox.org/)
%--------------------------------------------------------------------------

clc;clear;close all;
addpath(genpath('.'));

%% Generate coupled data
X_dims = [100, 50, 500];
Y_dims = [500, 20];
D_X    = length(X_dims);
D_Y    = length(Y_dims);
tao    = round(0.2 * X_dims(end));
TT     = X_dims(end) - tao;
R      = 10;
[X, Y] = generate_data(X_dims, Y_dims, R, 20, "MT");
K_new = 20;
unfold_N = X_dims(1) * X_dims(2);

%% Initialize factor matrices
idxX = repmat({':'}, 1, D_X);
idxX(end) = {1:tao}; 
initX = X(idxX{:}); 
idxY = repmat({':'}, 1, D_Y);  
idxY(1) = {1:tao};
initY = Y(idxY{:});

[tensor_cp, matrix_pred, rec_errors] = cmtf_MT(initX, initY, 5, 'svd', 500, 1e-6); 
initAs = tensor_cp.factors;
initAs{1} = initAs{1} * diag(tensor_cp.lambda);
tensor_cp.factors = initAs;

A = tensor_cp.factors{1};
B = tensor_cp.factors{2};
C = tensor_cp.factors{3};
D = matrix_pred.factors{2};

AT = A';
ATA = AT * A; 
BT = B';
BTB = BT * B;
CT = C';
CTC = CT * C;
DT = D';
DTD = DT * D;
Call = C;

%% Initialize auxiliary matrices
AP = double(tenmat(initX, 1)) * khatrirao(C, B);
AQ = CTC .* BTB;
BP = double(tenmat(initX, 2)) * khatrirao(C, A);
BQ = CTC .* ATA;
DP = initY' * C;
DQ = CTC;

%% Initialize frequency filter
phi_X = zeros(1, K_new);
phi_Y = zeros(1, K_new);

%% Update
n_iter = 10; % number of iteration
lambda_ = 0.9; % initial forgetting factor for tensor
mu_ = 0.9; % initial forgetting factor for matrix
eta_ = 0.05;  % learning rate
clip = @(x) max(0.5, min(1.0, x));
average_local_error_x = 0.0;
average_local_error_y = 0.0;
k = 0;

num_windows = fix(double(TT) / double(K_new));
result_running_time = zeros(1, num_windows);
result_local = zeros(1, num_windows);
result_global = zeros(1, num_windows); 
result_forgetting_factors_x = zeros(1, num_windows); 
result_forgetting_factors_y = zeros(1, num_windows); 

for t = 1 : K_new : TT
    if tao + t + K_new - 1 > X_dims(end)
        break
    end
    fprintf('[%dth update]\n', k+1);

    % get incoming data
    endTime = min(tao + t + K_new - 1, X_dims(end));
    idxX(end) = {tao + t:endTime};
    idxY(1) = {tao + t:endTime};
    Xnew = squeeze(X(idxX{:}));
    Ynew = squeeze(Y(idxY{:}));
    unfolded3 = double(tenmat(Xnew, 3));
    unfolded2 = double(tenmat(Xnew, 2));
    unfolded1 = double(tenmat(Xnew, 1));

    % get accumulated data
    idxX(end) = {1:endTime};
    Xacc = tensor(X(idxX{:}));

    start_time = tic();

    % update frequency filter
    fx = fft(reshape(single(Xnew), unfold_N, K_new), [], 2);
    fy = fft(Ynew, [], 1)';
    phi_X = lambda_ * phi_X + sum(abs(fx), 1)/unfold_N; 
    phi_Y = mu_ * phi_Y + sum(abs(fy), 1)/Y_dims(2); 
    phi = exp(-2 * phi_X / norm(phi_X) - 2 * phi_Y / norm(phi_Y));
    phi_hat = fft(phi);
    G = toeplitz([phi_hat(1), phi_hat(end:-1:2)], phi_hat);
    G = real(G); 

    %  ALS 
    for iteration = 1 : n_iter
        % Upadte C
        kr = khatrirao(B, A);
        ATA = AT * A;
        DTD = DT * D;
        F = BTB .* ATA + DTD;
        if (iteration <= 1)
            C = sylvester(G, F, unfolded3 * kr + Ynew * D);
            CT = C';
        else
            CT = F \ (unfolded3 * kr + Ynew * D)';
            C = CT';
        end
    
        % Upadte D
        CTC = CT * C; 
        DP = mu_ * DP + Ynew' * C;
        DQ = mu_ * DQ + CTC;
        DT = DQ \ DP';
        D = DT';
    
        % Upadte B
        kr = khatrirao(C, A);
        F = CTC .* ATA;
        BP = lambda_ * BP + unfolded2 * kr;
        BQ = lambda_ * BQ + F;
        BT = BQ \ BP';
        B = BT';
        
        % Upadte A
        kr = khatrirao(C, B);
        BTB = BT * B;
        F = CTC .* BTB;
        AP = lambda_ * AP + unfolded1 * kr;
        AQ = lambda_ * AQ + F;
        AT = AQ \ AP';
        A = AT';
    end

    elapsed_time = toc(start_time);

    % Local error of X
    New_K_tensor = ktensor({A, B, C});
    New_T_tensor = full(New_K_tensor);
    new_diff = New_T_tensor - Xnew;
    new_normX = norm(new_diff(:)) / norm(Xnew(:));

    % Global error of X
    Call = cat(1, Call, C);
    K_tensor = ktensor({A, B, Call});
    T_tensor = full(K_tensor); 
    diffX = T_tensor - Xacc;
    normX = norm(diffX(:)) / norm(Xacc);  

    % Local error f Y
    new_K_matrix = ktensor({C, D});
    new_T_matrix = full(new_K_matrix); 
    new_diffY = new_T_matrix - Ynew;
    new_normY = norm(new_diffY(:)) / norm(Ynew(:));

    % Update forgetting factors
    average_local_error_x = (k * average_local_error_x + new_normX) / (k+1); 
    average_local_error_y = (k * average_local_error_y + new_normY) / (k+1); 
    lambda_ = clip(lambda_ + eta_ * (average_local_error_x - new_normX));
    mu_ = clip(mu_ + eta_ * (average_local_error_y - new_normY));
    k = k + 1;

    result_running_time(k) = elapsed_time;
    result_local(k) = new_normX;
    result_global(k) = normX;
    result_forgetting_factors_x(k) = lambda_;
    result_forgetting_factors_y(k) = mu_;

    fprintf('Running time: %.6f (sec) \t| ', elapsed_time);
    fprintf('Local error: %.6f \t| ', new_normX);
    fprintf('Global error: %.6f\n', normX);
end

disp('Finished');