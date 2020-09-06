class ModelRunNotCompletedError(Exception):
    def __init__(self, *args, **kwargs):
        self.message = "This model has not finished running. Please wait for the model to finish running before using " \
                       "an export method."
