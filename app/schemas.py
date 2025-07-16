from pydantic import BaseModel
from numpydantic import NDArray, Shape


class SegmentationResponse(BaseModel):
    frame_index: int
    mask_points: NDArray[Shape["*,2"], int]  # type: ignore
