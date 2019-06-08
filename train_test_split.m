function [train_idxs, test_idxs] = ...
    train_test_split(n_samples, train_size, random_state, varargin)

p = inputParser;
addRequired(p, 'n_samples')
addRequired(p, 'train_size')
addRequired(p, 'random_state')
addParameter(p, 'stratify', false)

parse(p, n_samples, train_size, random_state, varargin{:})

n_train = floor(n_samples * train_size);
n_test = n_samples - n_train;

% Still need to implement stratification
train_idxs = datasample(random_state, 1:n_samples, n_train);

% test_idxs are the complement of train_idxs
test_idxs = setdiff(1:n_samples, train_idxs);

end