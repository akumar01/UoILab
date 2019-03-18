classdef AbstractUoILinearModel
    
    properties
        selection_frac
        n_boots_sel
        n_boots_est
        stability_selection
        random_state
        selection_thresholds_
        n_supports
        estimation_score
        coef_
        reg_params_
        n_reg_params_
        supports_
        n_supports_
    end
    
    methods
        % Constructor
        function self = AbstractUoILinearModel(varargin)
            % Stupid workaround to deal with consecutive passes 
            % of varargin
            if nargin == 1 && isempty(varargin{1})
                varargin = {};
            end

            p = inputParser;
            addOptional(p, 'selection_frac', 0.9)
            addOptional(p, 'n_boots_sel', 48)
            addOptional(p, 'n_boots_est', 48)
            addOptional(p, 'stability_selection', 1)
            addOptional(p, 'random_state', NaN)
            parse(p, varargin{:})
            
            % Copy input arguments to object
            for fn = fieldnames(p.Results)'
                self.(fn{1}) = p.Results.(fn{1});
            end

            if isinteger(self.random_state)
               self.random_state = RandStream('mt19937ar', 'Seed',...
                                                self.random_state);
               RandStream.setGlobalStream(sef.random_state)
            end
            
            self.selection_thresholds_ = ...
            stability_selection_to_threshold(self.stability_selection,...
            self.n_boots_sel);
            
            self.n_supports = NaN;
        end

%         function self = get_n_coef(self, X, y)
%         
%         end
% 
%         function self = get_reg_params(self)
% 
%         end
% 
%         function self = score_predictions(self, y_true, y_pred, supports)
%         
%         end
% 
%         function self = intersect(self, coef, thresholds)
% 
%         end
%       
%         function selection_lm(self, varargin)
% 
%         end
%         
%         function estimation_lm(self, varargin)
%             
%         end
        
        function self = fit(self, X, y, varargin)
            % Flatten varargin
            varargin = varargin{1};

            p = inputParser;
            addOptional(p, 'stratify', NaN);
            addOptional(p, 'verbose', false);
            parse(p, varargin{:})
            stratify = p.Results.stratify;
            verbose = p.Results.verbose;
            [n_samples, n_coef] = self.get_n_coef(X, y);
            n_features = size(X, 2);
            
            % Selection module
            self.reg_params_ = self.get_reg_params(X, y);
            self.n_reg_params_ = length(self.reg_params_);
            
            selection_coefs = zeros(self.n_boots_sel, self.n_reg_params_,...
                                    n_coef);
            % Iterate over bootstraps
            for bootstrap = 1:self.n_boots_sel
                % draw a resamples bootstrap (need to re-implement to
                % support stratification!) 
                [trainInd, testInd] = dividerand(n_samples, ...
                    self.selection_frac, 0, 1 - self.selection_frac);
                X_rep = X(trainInd, :);
                y_rep = y(trainInd);
                
                for idx = 1:length(self.reg_params_)
                    selection_coefs(bootstrap, idx, :) = ...
                    self.selection_lm(X_rep, y_rep, self.reg_params_(idx));                    
                end
            end
            
            self.supports_ = self.intersect(selection_coefs,...
                                            self.selection_thresholds_);
            self.n_supports_ = size(self.supports_, 1);
            
            % Estimation Module
            
            % coefs_ for each bootstrap for each support
            self.estimates_ = zeros(self.n_boots_est, self.n_supports,...
                n_coef);
            
            % score (r2/AIC/AIC/AICc/BIC) for each bootstrap for each
            % support
            self.scores_ = zeros(self.n_boots_est, self.n_supports_);
            
            n_tiles = floor(n_coef/n_features);
            
            % Iterate over bootstrap samples
            for bootstrap = 1:self.n_boots_est
                % Draw a resampled bootstrap
                [X_train, X_test, y_train, y_test] = train_test_split(X, y,...
                    'test_size', 1 - self.selection_frac, 'stratify',...
                    stratify, 'random_state', self.random_state);
                % Iterate over regularization parameters
                for idx = 1:length(self.supports_)
                   support = self.supports_(idx);
                   % Extract current support set
                   % If nothing was selected, do not bother running OLS
                   if isempty(support)
                      y_pred = zeros(numel(y_test));
                   else
                       % Compute ols estimate and store the fitted
                       % coefficients
                       coefs = self.estimation_lm(X_train(:, support),...
                           y_train);
                       self.estimates_(...
                       bootstrap, idx, repmat(support, n_tiles)) = coefs;
                       % Obtain predictions for scoring
                       y_pred = self.predict(coefs, X_test);
                   end
                   
                   self.scores_(bootstrap, idx) = self.score_predictions(...
                       self.estimation-score, y_test, y_pred, support);
                end
            
                [~, self.rp_max_idx] = max(self.scores, [], 2);
                % extract the estimates over bootstraps from model with
                % best regularization parameter value
                best_estimates = self.estimates_(1:self.n_boots_est,...
                                                self.rp_max_idx, :);
                self.coef_ = reshape(median(best_estimates, 1), n_tile,...
                    n_features);
            end
            
        end
        
    end
        
        
        
end

