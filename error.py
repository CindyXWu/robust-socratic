class FuncInputError(Exception):
    """Class for passing wrong parameters"""
    def __init__(self):
        print("Wrong parameters passed.")