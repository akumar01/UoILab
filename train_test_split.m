function [train_idxs, test_idxs] = ...
    train_test_split(n_samples, train_size, varargin)

p = inputParser;
addRequired(p, 'n_samples')
addRequired(p, 'train_size')
addParameter(p, 'shuffle', true)
addParameter(p, 'stratify', false)

parse(p, n_samples, train_size, varargin{:})

n_train = floor(n_samples * train_size);
n_test = n_samples - n_train;

% Still need to implement stratification
train_idxs = datasample(1:n_samples, n_train);

% test_idxs are the complement of train_idxs
test_idxs = setdiff(1:n_samples, train_idxs);

end