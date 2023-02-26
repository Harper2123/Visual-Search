def mapping(feature_dict, reduced_feat_vector):
    """
    Maps the reduced feature vector to the corresponding image file name.

    Args:
    - feature_dict (dict): A dictionary containing image file names as keys and feature vectors as values.
    - reduced_feat_vector (list): A list of reduced feature vectors.

    Returns:
    - feature_dict (dict): A dictionary containing image file names as keys and the corresponding reduced feature vectors as values.
    """
    # Loop through each reduced feature vector and its corresponding key in the feature_dict
    for reduced_vector, key in zip(reduced_feat_vector, feature_dict.keys()):
        # Map the reduced feature vector to the key in the feature_dict
        feature_dict[key] = reduced_vector
    # Return the updated feature_dict
    return feature_dict
