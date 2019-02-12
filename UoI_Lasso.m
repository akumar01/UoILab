classdef UoI_Lasso < AbstractUoILinearRegressor
    properties
        n_lambdas
        eps
        warm_start
    end
   
    methods
        function self = UoI_Lasso(varargin)
            self = self@AbstractUoILinearRegressor(varargin);
            
            p = inputParser;
            addOptional(p, 'n_lambdas', 48)
            addOptional(p, 'eps', 0.001)
            addOptional(p, 'warm_start', true)
            parse(p, varargin{:})
            
            % Copy input arguments to object
            for fn = fieldnames(p.Results)'
                self.(fn{1}) = p.Results.(fn{1});
            end            
        end
        
        % Handle call to matlab Lasso function
        function coefs = selection_lm(self, X, y, reg_params)
       
            coefs = lasso(X, y, 'alpha', 1, 'lambda', reg_params.lambda,...
                  'AbsTol', self.tol, 'MaxIter', self.max_iter);
  
        end
        
       % Handle call to matlab OLS
        function coefs = estimation_lm(X, y)
            
            coefs = mvregress(X, y);
            
        end

        % Reproduce behavior of sklearn's alpha grid
        function reg_params = get_reg_parmas(X, y, varargin)
        
            p = inputParser;
            addOptional(p, 'l1_ratio', 0.5)
            addOptional(p, 'n_boots_sel', 48)
            addOptional(p, 'n_boots_est', 48)
            addOptional(p, 'stability_selection', 1)
            addOptional(p, 'random_state', NaN)
            parse(p, varargin{:})

            
            
    if l1_ratio == 0:
        raise ValueError("Automatic alpha grid generation is not supported for"
                         " l1_ratio=0. Please supply a grid by providing "
                         "your estimator with the appropriate `alphas=` "
                         "argument.")
    n_samples = len(y)

    sparse_center = False
    if Xy is None:
        X_sparse = sparse.isspmatrix(X)
        sparse_center = X_sparse and (fit_intercept or normalize)
        X = check_array(X, 'csc',
                        copy=(copy_X and fit_intercept and not X_sparse))
        if not X_sparse:
            # X can be touched inplace thanks to the above line
            X, y, _, _, _ = _preprocess_data(X, y, fit_intercept,
                                             normalize, copy=False)
        Xy = safe_sparse_dot(X.T, y, dense_output=True)

        if sparse_center:
            # Workaround to find alpha_max for sparse matrices.
            # since we should not destroy the sparsity of such matrices.
            _, _, X_offset, _, X_scale = _preprocess_data(X, y, fit_intercept,
                                                          normalize,
                                                          return_mean=True)
            mean_dot = X_offset * np.sum(y)

    if Xy.ndim == 1:
        Xy = Xy[:, np.newaxis]

    if sparse_center:
        if fit_intercept:
            Xy -= mean_dot[:, np.newaxis]
        if normalize:
            Xy /= X_scale[:, np.newaxis]

    alpha_max = (np.sqrt(np.sum(Xy ** 2, axis=1)).max() /
                 (n_samples * l1_ratio))

    if alpha_max <= np.finfo(float).resolution:
        alphas = np.empty(n_alphas)
        alphas.fill(np.finfo(float).resolution)
        return alphas

    return np.logspace(np.log10(alpha_max * eps), np.log10(alpha_max),
                       num=n_alphas)[::-1]
        end
    end
   
end