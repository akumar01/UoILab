function run_tests

test_stability_selection_to_threshold()
test_stability_selection_to_threshold_float()
test_stability_selection_to_threshold_ints()
test_stability_selection_to_threshold_floats()
test_stability_selection_to_threshold_exceeds_n_bootstraps()
test_stabiliy_to_threshold_one_bootstrap()
test_stability_selection_to_threshold_input_value_error()
test_stability_selection_reject_negative_numbers()
test_intersection()
test_intersection_with_stability_selection_one_threshold()
test_intersection_with_stability_selection_multiple_thresholds()
test_intersection_no_thresholds()

end

function test_stability_selection_to_threshold
% Tests whether stability_selection_to_threshold correctly
% outputs the correct threshold when provided a single integer.

n_boots_sel = 48;
test_int = 36;


selection_thresholds = ...
    stability_selection_to_threshold(test_int, n_boots_sel);

assert(selection_thresholds == 36)

end

function test_stability_selection_to_threshold_float
% Tests whether stability selection_to_threshold correctly
% outputs the correct threshold when provided a single float

n_boots_sel = 48;
test_float = 0.5;
selection_thresholds = ...
    stability_selection_to_threshold(test_float, n_boots_sel);

assert(selection_thresholds == 24)

end


function test_stability_selection_to_threshold_ints

n_boots_sel = 48;
test_ints = [24, 28, 33, 38, 43, 48];
selection_thresholds = ...
    stability_selection_to_threshold(test_ints, n_boots_sel);
assert(isequal(selection_thresholds, [24, 28, 33, 38, 43, 48]))

end

function test_stability_selection_to_threshold_floats

n_boots_sel = 48;
test_floats = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

selection_thresholds = ...
    stability_selection_to_threshold(test_floats, n_boots_sel);

assert(isequal(selection_thresholds, [24, 28, 33, 38, 43, 48]))

end

function test_stability_selection_to_threshold_exceeds_n_bootstraps

n_boots_sel = 48;
test_floats = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1];
test_ints = [24, 28, 33, 38, 43, 48, 52];

try
    stability_selection_to_threshold(test_ints, n_boots_sel)
    error('Stability selection did not fail on test ints')
catch

end

try
    stability_selection_to_threshold(test_floats, n_boots_sel)
    error('Stability selection did not fail on test floats')
catch

end

end

function test_stabiliy_to_threshold_one_bootstrap
% Tests edge case where one bootstrap is requested

n_boots_sel = 1;

threshold = 1;

selection_thresholds = ...
    stability_selection_to_threshold(n_boots_sel, threshold);

assert(isequal(selection_thresholds, 1))

end

function test_stability_selection_to_threshold_input_value_error
% Tests whether stability_selection_to_threshold properly raises an error
% when it receives objects without ints or floats

n_boots_sel = 48;
stability_selection_list = [0, 1, 'a'];
stability_selection_cell = {'a', 'b'};
stability_selection_string = 'hello';

try
    stability_selection_to_threshold(stability_selection_list, n_boots_sel);
    error('stability_selection did not raise error on stability_selection_list')
catch    
end 

try
    stability_selection_to_threshold(stability_selection_cell, n_boots_sel);
    error('stability_selection did not raise error on stability_selection_cell')
catch    
end 

try
    stability_selection_to_threshold(stability_selection_string, n_boots_sel);
    error('stability_selection did not raise error on stability_selection_string')
catch    
end 

end

function test_stability_selection_reject_negative_numbers
% Tests whether stability selection to threshold correctly rejects negative
% thresholds

n_boots_sel = 48;

test_negative = -1 * [24, 28, 33, 43, 48, 52];

try
    stability_selection_to_threshold(test_negative, n_boots_sel);
    error('stability_selection did not raise error on negative numbers')
catch    
end 


end

function test_intersection
% Tests hard intersection

coefs = [2, 1, -1, 0, 4;...
      4, 0, 2, -1, 5;...
      1, 2, 3, 4, 5]';
coefs(:, :, 2) = [2, 0, 0, 0, 0;...
                   3, 1, 1, 0, 3;...
                   6, 7, 8, 9, 10]';
coefs(:, :, 3) = [2, 0, 0, 0, 0;...
                  2, -1, 3, 0, 2;...
                  2, 4, 6, 8, 9]';

% Re-order the axes
coefs = permute(coefs, [3, 2, 1]);
              
true_intersection = [true, false, false, false, false;
                     true, false, true, false, true;
                     true, true, true, true, true];

selection_thresholds = 3;

estimated_intersection = ...
        intersection(coefs, selection_thresholds);

true_intersection = sort(true_intersection, 1);
estimated_intersection = sort(logical(estimated_intersection), 1);
    
assert(isequal(true_intersection, estimated_intersection))
    
end

function test_intersection_with_stability_selection_one_threshold
% Tests that the intersection method correctly calculates the intersection
% using the number of bootstraps as the default selectiom threshold

coefs = [2, 1, -1, 0, 4;...
      4, 0, 2, -1, 5;...
      1, 2, 3, 4, 5]';
coefs(:, :, 2) = [2, 0, 0, 0, 0;...
                   3, 1, 1, 0, 3;...
                   6, 7, 8, 9, 10]';
coefs(:, :, 3) = [2, 0, 0, 0, 0;...
                  2, -1, 3, 0, 2;...
                  2, 4, 6, 8, 9]';

% Re-order the axes
coefs = permute(coefs, [3, 2, 1]);

true_intersection = [true, false, false, false, false;
                     true, true, true, false, true;
                     true, true, true, true, true];

selection_threshold = 2;

estimated_intersection = intersection(coefs, selection_threshold);

true_intersection = sort(true_intersection, 1);
estimated_intersection = sort(logical(estimated_intersection), 1);
    
assert(isequal(true_intersection, estimated_intersection))

end

function test_intersection_with_stability_selection_multiple_thresholds
% Tests whether intersection correctly performs an intersection with
% multiple thresholds. This test also covers the case when there are
% duplicates.

coefs = [2, 1, -1, 0, 4;...
      4, 0, 2, -1, 5;...
      1, 2, 3, 4, 5]';
coefs(:, :, 2) = [2, 0, 0, 0, 0;...
                   3, 1, 1, 0, 3;...
                   6, 7, 8, 9, 10]';
coefs(:, :, 3) = [2, 0, 0, 0, 0;...
                  2, -1, 3, 0, 2;...
                  2, 4, 6, 8, 9]';

% Re-order the axes
coefs = permute(coefs, [3, 2, 1]);

true_intersection = [true, false, false, false, false;
                     true, true, true, false, true;
                     true, true, true, true, true;
                     true, false, true, false, true];

selection_thresholds = [2, 3];

estimated_intersection = intersection(coefs, selection_thresholds);

true_intersection = sort(true_intersection, 1);
estimated_intersection = sort(logical(estimated_intersection), 1);
    
assert(isequal(true_intersection, estimated_intersection))

end

function test_intersection_no_thresholds
% Tests that the intersection correctly calculates the intersection using
% the number of bootstraps as the default selection threshold

coefs = [2, 1, -1, 0, 4;...
      4, 0, 2, -1, 5;...
      1, 2, 3, 4, 5]';
coefs(:, :, 2) = [2, 0, 0, 0, 0;...
                   3, 1, 1, 0, 3;...
                   6, 7, 8, 9, 10]';
coefs(:, :, 3) = [2, 0, 0, 0, 0;...
                  2, -1, 3, 0, 2;...
                  2, 4, 6, 8, 9]';

% Re-order the axes
coefs = permute(coefs, [3, 2, 1]);

true_intersection = [true, false, false, false, false;
                     true, false, true, false, true;
                     true, true, true, true, true];

estimated_intersection = intersection(coefs);

true_intersection = sort(true_intersection, 1);
estimated_intersection = sort(logical(estimated_intersection), 1);
    
assert(isequal(true_intersection, estimated_intersection))


end
