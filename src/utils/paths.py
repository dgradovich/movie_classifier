import os

def  set_main_path():
    """
    Find main path to the project

    :return: Main path to the project
    """

    return os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )
    )


def get_data_path(data_filename):
    """
    Get the absolute path to data

    :return: Absolute path to data 
    """
    project_path = set_main_path()
    return os.path.join(project_path,
        'src',
        'resources',
        'data', 
        'the-movies-dataset',
        data_filename)


def get_models_path(model_object_filename):
    """
    Get the absolute path to models and vectorizer objects

    :return: Absolute path to data 
    """

    project_path = set_main_path()
    return os.path.join(project_path,
        'src',
        'resources',
        'models',
        model_object_filename)


def get_output_path(model_object_filename):
    """
    Get the absolute path to models and vectorizer objects

    :return: Absolute path to data 
    """

    project_path = set_main_path()
    return os.path.join(project_path,
        'src',
        'resources',
        'output',
        model_object_filename)
