class Frame:
    def __init__(self,frame_id,img):
        self.frame_id = frame_id
        self.img = img
        self.detections = []

    def add_detection(self,detection):
        self.detections.append(detection)

