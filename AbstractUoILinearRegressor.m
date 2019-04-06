classdef AbstractUoILinearRegressor < AbstractUoILinearModel

    properties
        fit_intercept
        normalize
        max_iter
        intercept_
    end
    
    methods
        
        % Constructor
        function self = AbstractUoILinearRegressor(varargin)
            % Stupid workaround to deal with consecutive passes 
            % of varargin
            if nargin == 1 && isempty(varargin{1})
                varargin = {};
            end
            self = self@AbstractUoILinearModel(varargin);
                        
        end

        function [n_samples, n_coef] = get_n_coef(self, X, y)
            [n_samples, n_coef] = size(X);
        end
        
        function score = score_predictions(self, metric, y_true, y_pred,...
                supports)
                        
            if strcmp(metric, 'r2')
                score = ESF.r2score(y_true, y_pred);
            else
                n_features = nnz(supports);
                n_samples = numel(y_true);
                ll = log_likelihood_glm('normal', y_ytrue, y_pred);
                if strcmp(metric, 'BIC')
                    score = ESF.BIC(ll, n_features, n_samples);
                elseif strcmp(metric, 'AIC')
                    score = ESF.AIC(ll, n_features);
                elseif strcmp(metric, 'AICc')
                    score = ESF.AICc(ll, n_features, n_samples);
                else
                    error('%s is not a valid option', metric)
                end
                score = -score;
            end
            
        end

        function intsct = intersect(self, coef, thresholds)
            intsct = intersection(coef, thresholds);
        end
                   
        function self = fit(self, X, y, varargin)
            % Flatten varargin
            varargin = varargin{1};
            [X, y] = check_X_y(X, y);
            [X, y, X_offset, y_offset, X_scale] = ...
                preprocess_data(X, y, self.fit_intercept, self.normalize);
            self = fit@AbstractUoILinearModel(self, X, y, varargin);
            self = self.set_intercept(X_offset, y_offset, X_scale);
            self.coef_ = squeeze(self.coef_);
        end
        
        function self = set_intercept(self, X_offset, y_offset, X_scale)
        % Set the intercept
            if self.fit_intercept
                self.coef_ = self.coef_ ./ X_scale;
                self.intercept_ = y_offset - dot(X_offset, self.coef_');            
            else
                self.intercept_ = 0;
            end
        end
        
        function y_pred = predict(self, X, coef)
           if nargin < 3
               coef = self.coef_;
           end
           
           y_pred = X * coef;
           
        end
        
    end
    
end