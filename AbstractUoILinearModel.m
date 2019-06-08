classdef AbstractUoILinearModel
    
    properties
        selection_frac
        estimation_frac
        n_boots_sel
        n_boots_est
        stability_selection
        random_state
        selection_thresholds_
        n_supports
        est_score
        coef_
        reg_params_
        n_reg_params_
        supports_
        estimates_
        scores_
        n_supports_
        rp_max_idx
    end
    
    methods
        % Constructor
        function self = AbstractUoILinearModel(varargin)
            
            % Package arguments
            args_struct = cell2struct({varargin{2:2:length(varargin)}},...
                                       {varargin{1:2:length(varargin)}}, 2);

            
            % Copy input arguments to object
            arg_fields = fields(args_struct);
            for i = 1:length(arg_fields)
               self.(arg_fields{i}) = args_struct.(arg_fields{i});
            end
                        
            if isnan(self.random_state)
                rng('shuffle')
            else
                global_stream = RandStream('mt19937ar', 'Seed',...
                                            self.random_state);
                RandStream.setGlobalStream(global_stream)
            end
            
            self.random_state = RandStream.getGlobalStream;
            
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
            addParameter(p, 'stratify', NaN);
            addParameter(p, 'verbose', false);
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
                [trainInd, ~] = train_test_split(n_samples, ...
                    self.selection_frac, self.random_state);
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
            self.estimates_ = zeros(self.n_boots_est, self.n_supports_,...
                n_coef);
            
            % score (r2/AIC/AIC/AICc/BIC) for each bootstrap for each
            % support
            self.scores_ = zeros(self.n_boots_est, self.n_supports_);
            
            n_tile = floor(n_coef/n_features);
            
            % Iterate over bootstrap samples
            for bootstrap = 1:self.n_boots_est
                % Draw a resampled bootstrap
                [trainInd, testInd] = train_test_split(n_samples, ...
                self.estimation_frac, self.random_state);
                X_train = X(trainInd, :);
                y_train = y(trainInd);
                X_test = X(testInd, :);
                y_test = y(testInd, :);

                % Iterate over regularization parameters
                for idx = 1:self.n_supports_
                   support = self.supports_(idx, :);
                   % Extract current support set
                   % If nothing was selected, do not bother running OLS
                   if isempty(nonzeros(support))
                      y_pred = zeros(numel(y_test), 1);
                   else
                       % Compute ols estimate and store the fitted
                       % coefficients
                       n_boot_samples = size(X_train, 1);
                       support_mask = logical(repmat(support, n_boot_samples, 1));
                       coefs = self.estimation_lm(reshape(X_train(support_mask) ,...
                               n_boot_samples, []), y_train);
                       estimate = zeros(n_features, 1);
                       estimate(logical(support)) = coefs;
                       self.estimates_(...
                       bootstrap, idx, :) = estimate;
                       % Obtain predictions for scoring
                       y_pred = self.predict(X_test, estimate);
                   end
                   
                   self.scores_(bootstrap, idx) = self.score_predictions(...
                       self.est_score, y_test, y_pred, support);
                end
                [~, self.rp_max_idx] = max(self.scores_, [], 2);
                % extract the estimates over bootstraps from model with
                % best regularization parameter value
                best_estimates = zeros(self.n_boots_est, n_features);
                for i = 1:self.n_boots_est
                    best_estimates(i, :) = self.estimates_(i,...
                        self.rp_max_idx(i), :);
                end
                
                self.coef_ = reshape(median(best_estimates, 1), n_tile,...
                    n_features);                
            end
            
        end
        
    end
        
        
        
end

