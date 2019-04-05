function selection_thresholds =...
    stability_selection_to_threshold(stability_selection, n_boots)

    if all((stability_selection <= 1) & (stability_selection > 0))
       selection_thresholds = stability_selection * n_boots;       
    else     
        selection_thresholds = stability_selection;
    end
    
    % Ensure that the selection thresholds are within the permissible 
    % range
    
    selection_thresholds = floor(selection_thresholds);
    
    if (any(selection_thresholds > n_boots) ||...
            any(selection_thresholds < 1))
       error('Stability selection thresholds not within correct bounds') 
    end
    
end