function supports = intersection(coefs, varargin)

if nargin > 1
   selection_thresholds = varargin{1};
else
   selection_thresholds = [size(coefs, 1)];
end

n_selection_thresholds = numel(selection_thresholds);
n_reg_params = size(coefs, 1);
n_features = size(coefs, 2);
supports = zeros(n_selection_thresholds, n_reg_params, n_features);

for i = 1:numel(selection_thresholds)
   supports(i, :, :) = sum(coefs ~= 0, 1) > selection_thresholds(i); 
end

supports = squeeze(reshape(supports, n_selection_thresholds * ...
                    n_reg_params, n_features));

supports = unique(supports, 'rows');
                
end