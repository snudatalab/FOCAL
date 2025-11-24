%--------------------------------------------------------------------------
% DEMO_ST
% 
% This is an implementation of "Fast and Accurate Online Coupled 
% Matrix-Tensor Factorization via Frequency Regularization"
% This code is designed for a single-tensor streaming setting.
%
% To run the code, Tensor Toolbox is required.
% Brett W. Bader, Tamara G. Kolda and others. MATLAB Tensor Toolbox 
% Available online (https://www.tensortoolbox.org/)
%--------------------------------------------------------------------------

clc;clear;close all;
addpath(genpath('.'));

%% Generate coupled data
X_dims = [200, 50, 500];
Y_dims = [200, 20];
N      = length(X_dims);
tao    = round(0.2 * X_dims(end));
TT     = X_dims(end) - tao;
R      = 10;
[X, Y] = generate_data(X_dims, Y_dims, R, 20, "ST");
K_new  = 20;
unfold_N = X_dims(1) * X_dims(2);
 
%% Initialize factor matrices
idx = repmat({':'}, 1, N);
idx(end) = {1:tao};
initX = X(idx{:});

[tensor_cp, matrix_pred, rec_errors] = cmtf_ST(initX, Y, 5, 'svd', 500, 1e-6);
initAs = tensor_cp.factors;
initAs{end} = initAs{end} * diag(tensor_cp.lambda);
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
Call = C;

%% Initialize auxiliary matrices
AP = double(tenmat(initX, 1)) * khatrirao(C, B);
AQ = CTC .* BTB;
BP = double(tenmat(initX, 2)) * khatrirao(C, A);
BQ = CTC .* ATA;

%% Initialize frequency filter
phi_X = zeros(1, K_new);

%% Update
n_iter = 10;  % number of iteration
lambda_ = 0.9;  % initial forgetting factor for tensor
eta_ = 0.05;  % learning rate
clip = @(x) max(0.5, min(1.0, x));
average_local_error_X = 0.0;
k = 0;

num_windows = fix(double(TT) / double(K_new));
result_running_time = zeros(1, num_windows);
result_local = zeros(1, num_windows);
result_global = zeros(1, num_windows); 
result_forgetting_factors = zeros(1, num_windows); 

for t = 1 : K_new : TT
    if tao + t + K_new - 1 > X_dims(end)
        break
    end
    fprintf('[%dth update]\n', k+1);

    % get incoming data
    endTime = tao + t + K_new - 1;
    idx(end) = {tao + t:endTime};
    Xnew = squeeze(X(idx{:}));
    unfolded3 = double(tenmat(Xnew, 3));
    unfolded2 = double(tenmat(Xnew, 2));
    unfolded1 = double(tenmat(Xnew, 1));

    % get accumulated data
    idx(end) = {1:endTime};
    Xacc = tensor(X(idx{:}));

    start_time = tic();
    
    % update frequency filter
    fx = fft(reshape(single(Xnew), unfold_N, K_new), [], 2);
    phi_X = lambda_ * phi_X + sum(abs(fx), 1)/unfold_N; 
    phi = exp(-2 * phi_X / norm(phi_X));
    phi_hat = fft(phi);
    G = toeplitz([phi_hat(1), phi_hat(end:-1:2)], phi_hat);
    G = real(G); 
    
    % ALS 
    for iteration = 1 : n_iter
        % Upadte C
        kr = khatrirao(B, A);
        ATA = AT * A;
        F = BTB .* ATA;
        if (iteration <= 1)
            C = sylvester(G, F, unfolded3 * kr);
            CT = C';
        else
            CT = F \ (unfolded3 * kr)';
            C = CT';
        end

        % Upadte D
        DT = A \ Y;
        D = DT';

        % Upadte B
        kr = khatrirao(C, A);
        CTC = CT * C; 
        F = CTC .* ATA;
        BP = lambda_ * BP + unfolded2 * kr;
        BQ = lambda_ * BQ + F;
        BT = BQ \ BP';
        B = BT';
        
        % Upadte A
        kr = khatrirao(C, B);
        BTB = BT * B;
        DTD = DT * D;
        AP = lambda_ * AP + unfolded1 * kr;
        AQ = lambda_ * AQ + CTC .* BTB;
        ALeft = AP + Y * D;
        ARight = AQ + DTD;
        AT = ARight \ ALeft';
        A = AT';
    end

    elapsed_time = toc(start_time);
    
    % Local error
    New_K_tensor = ktensor({A, B, C});
    New_T_tensor = full(New_K_tensor);
    new_diff = New_T_tensor - Xnew;
    new_norm = norm(new_diff(:)) / norm(Xnew(:));
    
    % Global error
    Call = cat(1, Call, C);
    K_tensor = ktensor({A, B, Call});
    T_tensor = full(K_tensor); 
    diffX = T_tensor - Xacc;
    normX = norm(diffX(:)) / norm(Xacc);  

    % Update forgetting factor
    average_local_error_X = (k * average_local_error_X + new_norm) / (k+1); 
    lambda_ = clip(lambda_ + eta_ * (average_local_error_X - new_norm));
    k = k + 1;

    result_running_time(k) = elapsed_time;
    result_local(k) = new_norm;
    result_global(k) = normX;
    result_forgetting_factors(k) = lambda_;

    fprintf('Running time: %.6f (sec) \t| ', elapsed_time);
    fprintf('Local error: %.6f \t| ', new_norm);
    fprintf('Global error: %.6f\n', normX);
end

disp('Finished');