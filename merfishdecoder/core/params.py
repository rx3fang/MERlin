class merfishParams:
    """
    An object that contains input variables.
    """
    def __init__(self, **arguments):
        for (arg, val) in arguments.items():
            setattr(self, arg, val)
    
    def to_string(self):
        return ("\n".join(["%s = %s" % (str(key), str(val)) \
                for (key, val) in self.__dict__.items() ]))

