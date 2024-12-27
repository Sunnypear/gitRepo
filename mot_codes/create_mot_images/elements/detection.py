
class Detection:
    def __init__(self,tlwh,confidence=0.8):
        self.tlwh = tlwh
        self.confidence = confidence
