class NoTargetError(Exception):
    def __init__(self, *args, **kwargs):
        self.message = "The current agent does not have a target, therefore it cannot calculate target azimuth & " \
                       "distance. "
