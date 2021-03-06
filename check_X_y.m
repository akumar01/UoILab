% Ensure that X and y are corresponding sizes and contain the appropriate
% data types. The sklearn version of this function has a lot of
% functionality. Here, we implement take as immutable defaults the set of
% options used in PyUoI.
function [X, y] = check_X_y(X, y)

    if isscalar(X)
        error('X must be a matrix')
    end

    if isscalar(y)
       error('y must be a matrix') 
    end

    if isempty(y)
        error('y cannot be empty')
    end

    if isempty(X)
        error('X cannot be empty')
    end    

    % Ensure all data types are numeric
    if ~isnumeric(X)
       error('X must be numeric')
    end

    if ~isnumeric(y)
        error('y must be numeric')
    end
    
    % Force all values to be finite:   
    if sum(isnan(X(:))) > 0 || sum(isinf(X(:))) > 0
        error('X contains inf or nans')
    end

    % Force all values to be finite:   
    if sum(isnan(y(:))) > 0 || sum(isinf(y(:))) > 0
        error('y contains inf or nans')
    end
    
    % Make sure that X and y have consistent lengths
    if size(X, 1) ~= size(y, 1)
        error('X and y do not have consistent dimensions. First dimension must equal n_samples')
    end
end