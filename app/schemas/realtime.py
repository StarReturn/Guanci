from pydantic import BaseModel


class RealtimeSessionCreate(BaseModel):
    digital_human_id: str
    mode: str = "webrtc"


class RealtimeSessionRead(BaseModel):
    id: str
    digital_human_id: str
    avatar_engine: str
    render_mode: str
    state: str
    keyframe_package_required: bool
    signaling: dict

