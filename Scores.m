% Change name of this class
classdef Scores
    methods(Static)
        function ll = log_likelihood_glm(model, y_true, y_pred)
            if strcmp(model, 'normal')
                rss = (y_true - y_pred).^2;
                n_samples = numel(y_true);
                ll = -n_samples/(2 * (1 + log(mean(rss))));
            elseif strcmp(model, 'poisson')
                ll = mean(y_true * log(y_pred) - y_pred);
            else
                error('Model is not available')
            end
        end
            
        function s = r2score(y_true, y_pred)
            SSres = sum((y_true - y_pred).^2);
            SStot = sum((y_true - mean(y_true)).^2);
            s = 1 - SSres/SStot;
        end
       
        function s = BIC(ll, n_features, n_samples)
            
            s = n_features * log(n_samples) - 2 * ll;
        
        end
        
        function s = AICc(ll, n_features, n_samples)
            s = 2 * n_features - 2  *ll;
            if n_samples > (n_features + 1)
               s = s + 2 * (n_features^2 + n_features)/...
                   (n_samples - n_features - 1);
            end
        end

        function s = AIC(ll, n_features)
            s = 2 * n_features - 2 * ll;
        end

        
   end
end