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





end