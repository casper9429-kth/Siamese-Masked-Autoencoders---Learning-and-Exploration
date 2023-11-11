def get_obj_from_str(string):
    """Converts a string to an object.

    Args:
        string (str): A string of the form "module.submodule:object".

    Returns:
        object: The object corresponding to the string.

    Example:
        ``np_array = get_object_from_string("numpy:array")``

        equivalent to:
        
        ``np_array = numpy.array``        
    """
    module_name, object_name = string.split(":")
    module = __import__(module_name, fromlist=[object_name])
    return getattr(module, object_name)

