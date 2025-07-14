class SegmentationResult:
    frame_index: int
    
    def __init__(self, frame_number: int) -> None:
        self.frame_index = frame_number