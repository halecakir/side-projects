"""TODO"""


class BaseModel:
    """TODO"""

    def __init__(self, options):
        self.options = options

    def _cos_similarity(self, vec1, vec2):
        from numpy import dot
        from numpy.linalg import norm

        return dot(vec1, vec2) / (norm(vec1) * norm(vec2))
