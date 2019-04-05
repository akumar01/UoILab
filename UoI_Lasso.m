classdef UoI_Lasso < AbstractUoILinearRegressor
    properties
        n_lambdas
        tol
        warm_start
    end
   
    methods
        function self = UoI_Lasso(varargin)
            self = self@AbstractUoILinearRegressor;

            p = inputParser;
            addParameter(p, 'n_lambdas', 48)
            addParameter(p, 'tol', 0.001)
            addParameter(p, 'warm_start', true)
            parse(p, varargin{:})
               
            % Copy input arguments to object
            for fn = fieldnames(p.Results)'
                self.(fn{1}) = p.Results.(fn{1});
            end            
        end
        
        % Handle call to matlab Lasso function
        function coefs = selection_lm(self, X, y, reg_params)
       
            coefs = lasso(X, y, 'Alpha', 1, 'lambda', reg_params);
  
        end
        
       % Handle call to matlab OLS
        function coefs = estimation_lm(self, X, y)
            lm = fitlm(X, y, 'Intercept', false);
            coefs = lm.Coefficients.Estimate;
        end

        % Pass on fitting responsibilities to superclass
        function self = fit(self, X, y, varargin)
            self = fit@AbstractUoILinearRegressor(self, X, y, varargin);
        end
            
        % Reproduce behavior of sklearn's alpha grid
        function lambdas = get_reg_params(self, X, y, varargin)
        
            p = inputParser;
            addParameter(p, 'l1_ratio', 0.5)
            addParameter(p, 'n_boots_sel', 48)
            addParameter(p, 'n_boots_est', 48)
            addParameter(p, 'stability_selection', 1)
            addParameter(p, 'random_state', NaN)
            parse(p, varargin{:})
            
            n_samples = length(y);

            [X, y, X_offset, ~, X_scale] = preprocess_data(X, y, self.fit_intercept, ...
                            self.normalize);
            Xy = X' * y;

            mean_dot = X_offset * sum(y);

            % Unclear if we need this
            %     if ndims(Xy) == 1
            %         Xy = Xy[:, np.newaxis]
            %     end

            if self.fit_intercept
                Xy = Xy - mean_dot;
            end
            
            if self.normalize
                Xy = Xy./X_scale;
            end
            
            lambda_max = max(sqrt(sum(Xy.^2, 2)))/(n_samples);

            if lambda_max <= eps
                lambdas = self.eps * ones(self.n_lambdas, 1);
            else
                % Need to check this has the right shape
                lambdas = logspace(log10(lambda_max * eps), log10(lambda_max),...
                               self.n_lambdas);
            end
        end
    end
   
end