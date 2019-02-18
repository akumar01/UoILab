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
        function alphas = get_reg_parmas(X, y, varargin)
        
            p = inputParser;
            addOptional(p, 'l1_ratio', 0.5)
            addOptional(p, 'n_boots_sel', 48)
            addOptional(p, 'n_boots_est', 48)
            addOptional(p, 'stability_selection', 1)
            addOptional(p, 'random_state', NaN)
            parse(p, varargin{:})
            
            if l1_ratio == 0
                error({'Automatic alpha grid generation is not supported for l1_ratio=0. Please supply a grid by providing your estimator with the appropriate `alphas=` argument.'})
            end

            n_samples = length(y);

            % Need to implement check_array and preprocess_data
            [X, y, ~, ~, ~] = preprocess_data(X, y, fit_intercept,normalize,'copy', false);
            Xy = X' * y;

            mean_dot = X_offset * sum(y);

            % Unclear if we need this
            %     if ndims(Xy) == 1
            %         Xy = Xy[:, np.newaxis]
            %     end

            if fit_intercept
                Xy = Xy - mean_dot;
            end
            
            if normalize
                Xy = Xy./X_scale;
            end
            
            alpha_max = max(sqrt(sum(Xy.^2, 2)))/(n_samples * l1_ratio);

            if alpha_max <= eps
                alphas = eps * ones(n_alphas, 1);
            else
                % Need to check this has the right shape
                alphas = logspace(log10(alpha_max * eps), log10(alpha_max),...
                               n_alphas);
            end
        end
    end
   
end