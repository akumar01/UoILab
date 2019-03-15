function selection_thresholds =...
    stability_selection_to_threshold(stability_selection, n_boots)

    if isfloat(stability_selection)
       selection_thresholds = [int16(stability_selection * n_boots)];

    elseif isinteger(stability_selection)
        selection_thresholds = [in16(stability_selection)];
    
    elseif numel(stability_selection) > 1
        % List of floats
        if isa(stability_selection, 'double')
           selection_thresholds = n_boots * stability_selection; 
        end
    end
    
    % Ensure that the selection thresholds are within the permissible 
    % range
    
    selection_thresholds = int16(selection_thresholds);
    
    if (any(selection_thresholds > n_boots) ||...
            any(selection_thresholds < 1))
       error('Stability selection thresholds not within correct bounds') 
    end
    
end