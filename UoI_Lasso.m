classdef UoI_Lasso < AbstractUoILinearRegressor
    properties
        n_lambdas
        tol
        warm_start
        eps
    end
   
    methods
        function self = UoI_Lasso(varargin)
            p = inputParser;
                        
            % Abstract Linear Model Parameters
            addParameter(p, 'selection_frac', 0.9)
            addParameter(p, 'estimation_frac', 0.9)
            addParameter(p, 'n_boots_sel', 48)
            addParameter(p, 'n_boots_est', 48)
            addParameter(p, 'stability_selection', 1)
            addParameter(p, 'random_state', NaN)
            
            % UoI Lasso parameters
            addParameter(p, 'n_lambdas', 48)
            addParameter(p, 'tol', 0.001)
            addParameter(p, 'warm_start', true)
            addParameter(p, 'fit_intercept', true)
            addParameter(p, 'normalize', true)
            addParameter(p, 'max_iter', 1000)
            addParameter(p, 'eps', 0.001)
            addParameter(p, 'est_score', 'r2',...
                @(x) any(strcmp({'r2', 'AIC', 'AICc', 'BIC'}, x)))
            parse(p, varargin{:})
            % Unpack p.Results into a cell array
            c1 = fieldnames(p.Results);
            c2 = struct2cell(p.Results);
            % Interleave
            vargs = cell(length(c1) + length(c2), 1);
            for i = 1:length(c1)
               vargs{2*i - 1} = c1{i};
               vargs{2 * i} = c2{i};
            end
           
            self = self@AbstractUoILinearRegressor(vargs{:});
                        
        end
        
        % Handle call to matlab Lasso function
        function coefs = selection_lm(self, X, y, reg_params, normalize)
       
            coefs = lasso(X, y, 'Alpha', 1, 'Lambda', reg_params,...
                          'Standardize', true);
  
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
                    
            n_samples = length(y);

            [X, y, ~, ~, ~] = preprocess_data(X, y, self.fit_intercept, ...
                            self.normalize);
            Xy = X' * y;
            
            lambda_max = max(sqrt(sum(Xy.^2, 2)))/(n_samples);

            if lambda_max <= self.eps
                lambdas = self.eps * ones(self.n_lambdas, 1);
            else
                lambdas = logspace(log10(lambda_max * self.eps), log10(lambda_max),...
                               self.n_lambdas);
            end
                        
        end
    end
   
end