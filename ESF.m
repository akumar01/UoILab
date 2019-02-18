classdef ESF
    methods(Static)
        function s = r2score(y_true, y_pred)
            SSres = sum((y_true - y_pred).^2);
            SStot = sum((y_true - mean(y_true)).^2);
            s = 1 - SSres/SStot;
        end
       
        function s = BIC(y_true, y_pred, n_features)
            n_samples = numel(y_true);
            
            RSS = sum((y_true - y_pred).^2);
            
            s = n_samples * log(RSS/n_samples) + ...
                n_features * log(n_samples);
        end

       
        function s = AIC(y_true, y_pred, n_features)
            n_samples = numel(y_true);
            
            RSS = sum((y_true - y_pred).^2);
            s = n_samples * log(RSS/n_samples) +...
                n_features * 2;
        end

        function s = AICc(y_true, y_pred, n_features)
            n_samples = numel(y_true);
            
            RSS = sum((y_true - y_pred).^2);
            
            if n_samples - n_features - 1 == 0
               s = AIC(y_true, y_pred, n_features); 
            else
               s = n_samples * log(RSS/n_samples) + ...
                   n_features * 2 + 2 * ...
                   (n_features^2 + n_features)/(n_samples - n_features - 1);
            end
            
        end
        
   end
end