%--------------------------------------------------------------------------
% CMTF_MT
%
% Calculates a coupled matrix and tensor factorization of a 3rd order tensor
% (X) and a matrix (Y), which share a temporal factor.
%
%   X = [[lambda; A, B, C]]
%   Y = [[gamma; C, D]]
%
% We assume that the temporal mode factor C is shared.
%
% Input:
%   tensor_3d        - 3D Tensor, e.g. from the Tensor Toolbox (tensor object)
%   matrix_2d        - 2D matrix that is coupled in the first mode with tensor_3d
%   rankCP           - rank for the CP decomposition
%   initType         - initialization type (e.g., 'svd')
%   nIterMax         - maximum number of ALS iterations (default: 1000)
%   tol              - tolerance for convergence (default: 1e-6)
%   normalizeFactors - boolean, if true, factors are normalized at the end
%   verbose          - boolean, if true, prints iteration info
%
% Output:
%   tensor_cp   - CPTensor of the 3D tensor: [[lambda; A, B, C]]
%   matrix_pred - CPTensor of the 2D matrix: [[gamma; C, D]]
%   rec_errors  - array of reconstruction errors at each iteration
%--------------------------------------------------------------------------

function [tensor_cp, matrix_pred, rec_errors] = ...
    cmtf_MT(tensor_3d, matrix_2d, rankCP, ...
                                          initType, nIterMax, tol, ...
                                          normalizeFactors, verbose)

    if nargin < 4 || isempty(initType)
        initType = 'svd';
    end
    if nargin < 5 || isempty(nIterMax)
        nIterMax = 1000;
    end
    if nargin < 6 || isempty(tol)
        tol = 1e-6;
    end
    if nargin < 7 || isempty(normalizeFactors)
        normalizeFactors = false;
    end
    if nargin < 8 || isempty(verbose)
        verbose = true;
    end
    
    if verbose
        fprintf('coupled_matrix_tensor_3d_factorization...\n');
    end

    %---------------------------------------------
    % 1) CP initialization (3D tensor)
    %    For example: tensor_cp is stored in the form (lambda, {A, B, C})
    %---------------------------------------------
    tensor_cp = initialize_cp(tensor_3d, rankCP, initType); 

    %---------------------------------------------
    % 2) Create 'coupled_unfold' by concatenating the unfolding of the
    %    tensor's first mode with matrix_2d, then perform CP initialization
    %    based on it to initialize the factor C.
    %---------------------------------------------
    unfoldX_mode1 = double(tenmat(tensor_3d, 3));
    coupled_unfold = [unfoldX_mode1, matrix_2d];
    
    coupled_init = initialize_cp(coupled_unfold, rankCP, initType);  
    
    % Replace only the third mode factor
    tensor_cp.factors{3} = coupled_init.factors{1};

    % For storing reconstruction errors
    rec_errors = zeros(nIterMax,1);

    %---------------------------------------------
    % 3) ALS Iterations
    %---------------------------------------------
    for iter = 1:nIterMax
        % (a) Update D for the matrix mode
        C_factor = tensor_cp.factors{3};
        D_temp = C_factor \ matrix_2d;
        D = D_temp';
        
        % (b) Update each mode of the tensor
        nDim = ndims(tensor_3d);  % should be 3
        for jj = nDim:-1:1
            % Compute Khatri-Rao product of all factors except the jj-th
            kr = khatri_rao_except(tensor_cp.factors, jj); 
            % Unfold the tensor along mode jj
            X_unfolded = double(tenmat(tensor_3d, jj));
            
            % If it is the third mode (jj == 3), then concatenate with the matrix
            if jj == 3
                kr = [kr; D];
                X_unfolded = [X_unfolded, matrix_2d];
            end
            
            % Solve for factor_jj in: kr * factor_jj' = X_unfolded'
            factor_jj_temp = kr \ X_unfolded';
            factor_jj = factor_jj_temp';
            tensor_cp.factors{jj} = factor_jj;
        end
        
        % (c) Calculate error
        %     Tensor error: || X - [[lambda; A, B, C]] ||^2
        %     Matrix error: || Y - [[gamma; C, D]] ||^2
        %     Here, lambda and gamma are not updated separately; only factors are updated.
        X_pred = full(ktensor(tensor_cp.lambda(:), tensor_cp.factors{:}));
        Y_pred = tensor_cp.factors{3} * D';  % C * D' (size: I1 x J)
        
        errX = norm(double(tensor_3d) - X_pred)^2;  
        errY = norm(matrix_2d - Y_pred)^2;
        error_new = errX + errY;
        
        rec_errors(iter) = error_new;
        
        % (d) Intermediate output
        if verbose && mod(iter, 10) == 1
            fprintf('Iter = %d\t|\tError = %g\n', iter, error_new);
        end
        
        % (e) Convergence check
        if iter > 1
            rel_change = abs(error_new - rec_errors(iter-1)) / rec_errors(iter-1);
            if (rel_change <= tol) || (error_new < tol)
                if verbose
                    fprintf('Converged after %d iterations.\n', iter);
                end
                rec_errors = rec_errors(1:iter);
                break;
            end
        end
        
        if iter == nIterMax
            warning('Reached maximum iteration number without convergence.');
        end
    end

    %---------------------------------------------
    % 4) Results (CP for the matrix: C and D)
    %---------------------------------------------
    % matrix_pred: [[gamma; C, D]]
    % Here, the lambda/gamma scalars are handled as None or [] without separate computation.
    matrix_pred.lambda = [];
    matrix_pred.factors = cell(1,2);
    matrix_pred.factors{1} = tensor_cp.factors{3};  % C
    matrix_pred.factors{2} = D;                     % D
    
    if normalizeFactors
        tensor_cp = cp_normalize(tensor_cp);
        matrix_pred = cp_normalize(matrix_pred);
    end
end

%--------------------------------------------------------------------------
% khatri_rao_except: Computes the Khatri-Rao product of all factors except the idx-th
%--------------------------------------------------------------------------
function KR = khatri_rao_except(factors, idx)
% factors: cell array {A, B, C, ...}
% idx: integer (1-based)
    nFactors = numel(factors);
    others = setdiff(1:nFactors, idx);
    KR = factors{others(end)};
    for i = (length(others)-1):-1:1
        KR = khatrirao(KR, factors{others(i)});
    end
end
