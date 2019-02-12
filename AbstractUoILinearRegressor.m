classdef AbstractUoILinearRegressor < AbstractUoILinearModel

    properties
        fit_intercept
        normalize
        max_iter
    end
    
    methods
        
        % Constructor
        function self = AbstractUoILinearRegressor(varargin)
            self = self@AbstractUoILinearModel(varargin);

            p = inputParser;
            addOptional(p, 'fit_intercept', true)
            addOptional(p, 'normalize', true)
            addOptional(p, 'max_iter', 1000)
            addOptional(p, 'estimation_score', 'r2',...
                @(x) any(strcmp({'r2', 'AIC', 'AICc', 'BIC'}, x)))
            parse(p, varargin{:})
            
            % Copy input arguments to object
            for fn = fieldnames(p.Results)'
                self.(fn{1}) = p.Results.(fn{1});
            end
    
            
        end

        function n_coef = get_n_coef(self, X, y)
            n_coef = size(X);
        end
        
        function score = score_predictions(self, metric, y_true, y_pred,...
                supports)
                        
            if strcmp(metric, 'r2')
                score = ESF.r2score(y_true, y_pred);
            else
                n_features = nnz(supports);
                if strcmp(metric, 'BIC')
                    score = scorefn.BIC(y_true, y_pred, n_features);
                elseif strcmp(metric, 'AIC')
                    score = scorefn.AIC(y_true, y_pred, n_features);
                elseif strcmp(metric, 'AICc')
                    score = ESF.AICc(y_true, y_pred, n_features);
                else
                    error('%s is not a valid option', metric)
                end
                score = -score;
            end
            
        end

        function intsct = intersect(self, coef, thresholds)
            intsct = intersection(coef, thresholds);
        end
        
        function self = preprocess_data(self, X, y)
            
        end
            
        function self = fit(self, X, y, varargin)
            [X, y] = check_X_y(X, y);
            [X, y, X_offset, y_offset, X_scale] = ...
                self.preproces_data(X, y);
            self = fit@AbstractUoILinearModel(X, y, varargin);
            self.set_intercept(X_offset, y_offset, X_scale);
            self.coef_ = squeeze(self.coef_);
        end
        
    end
    
end