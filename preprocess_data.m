function [X, y, X_offset, y_offset, X_scale] =  preprocess_data(X, y, fit_intercept, normalize)
    %{
    Centers data to have mean zero along axis 0. If fit_intercept=False or if
    the X is a sparse matrix, no centering is done, but normalization can still
    be applied. The function returns the statistics necessary to reconstruct
    the input data, which are X_offset, y_offset, X_scale, such that the output
        X = (X - X_offset) / X_scale
    X_scale is the L2 norm of X - X_offset. If sample_weight is not None,
    then the weighted mean of X and y is zero, and not the mean itself. If
    return_mean=True, the mean, eventually weighted, is returned, independently
    of whether X was centered (option used for optimization with sparse data in
    coordinate_descend).
    This is here because nearly all linear models will want their data to be
    centered. This function also systematically makes y consistent with X.dtype
    %}
    
    % convert y to the same data type as x
    if ~isa(y, class(X))
        cast(y, 'like', X)
    end
                     
    if fit_intercept
        X_offset = mean(X, 1);
        X = X - X_offset;
        if normalize
            X_scale = vecnorm(X, 2, 1);
            X = bsxfun(@rdivide, X, X_scale);
        else
            X_scale = ones(size(X, 2));
        end
        y_offset = mean(y);
        y = y - y_offset;
    else
        X_offset = zeros(size(X, 2));
        X_scale = ones(size(X, 2));
        if ndims(y) == 1
            y_offset = 0;
        else
            y_offset = zeros(size(y, 2));
        end
    end
end