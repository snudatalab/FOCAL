%--------------------------------------------------------------------------
% INITIALIZE_CP
%
% Performs CP decomposition initialization for a given data (matrix or tensor).
% If initType is 'svd', SVD-based initialization is used.
%
% For a 2D matrix:
%    data is of size (I x J).
%    We perform one SVD, then:
%       data ~ U * S * V'
%    The initial factors are {U, V}, and lambda is set to diag(S).
%
% For an N-dimensional tensor (e.g., 3D):
%    We unfold the data along each mode n, perform SVD on that unfolding,
%    and take the top-rankCP left singular vectors as the factor for that mode.
%    The initial lambda is set to a vector of ones of length rankCP, or an average
%    of singular values if preferred.
%
% Input:
%   data    - 2D matrix or N-D tensor (Tensor Toolbox 'tensor' or numeric array)
%   rankCP  - rank of decomposition
%   initType- 'svd'
%
% Output:
%   cp_struct - structure with fields:
%               .lambda  : 1 x rankCP (e.g., all ones or derived from S)
%               .factors : cell(1, N), each factor is (size of mode n) x rankCP
%--------------------------------------------------------------------------

function cp_struct = initialize_cp(data, rankCP, initType)
    if nargin < 3 || isempty(initType)
        initType = 'svd';
    end
    
    cp_struct.lambda  = [];
    cp_struct.factors = {};
    
    %---------------------------------------------------------------------------------
    % Case 1: When data is 2-dimensional
    %---------------------------------------------------------------------------------
    dims = size(data);
    if length(dims) == 2
        if isa(data, 'tensor')
            matData = double(data);
        else
            matData = data;
        end
        
        switch lower(initType)
            case 'svd'
                [U, S, V] = svds(matData, rankCP);
                svals = diag(S);
                cp_struct.lambda  = svals';
                cp_struct.factors = {U, V};
                
            otherwise
                error('Only "svd" initialization is implemented in this example.');
        end
        
    %---------------------------------------------------------------------------------
    % Case 2: When data is a tensor with 3 or more dimensions
    %---------------------------------------------------------------------------------
    else
        if isa(data, 'tensor')
            tensorData = data;
        else
            tensorData = tensor(data);
        end
        
        N = ndims(tensorData);
        cp_struct.factors = cell(1, N);
        
        switch lower(initType)
            case 'svd'
                for n = 1:N
                    unfoldMat = tenmat(tensorData, n);
                    matData   = double(unfoldMat);
                    [U, ~, ~] = svds(matData, rankCP);
                    cp_struct.factors{n} = U;
                end
                cp_struct.lambda = ones(1, rankCP);
                
            otherwise
                error('Only "svd" initialization is implemented in this example.');
        end
    end
end
