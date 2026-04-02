import logging
import math
from typing import List

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import av
import cv2
import numpy as np
import streamlit as st

from streamlit_webrtc import (
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
    create_mix_track,
    create_process_track,
)

st.set_page_config(page_title="Video Chat App")

logger = logging.getLogger(__name__)


# 🎥 Video Processing
class OpenCVVideoProcessor(VideoProcessorBase):
    type: Literal["noop", "cartoon", "edges", "rotate"]

    def __init__(self):
        self.type = "noop"

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        if self.type == "cartoon":
            img_color = cv2.pyrDown(cv2.pyrDown(img))
            for _ in range(6):
                img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
            img_color = cv2.pyrUp(cv2.pyrUp(img_color))

            img_edges = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img_edges = cv2.adaptiveThreshold(
                cv2.medianBlur(img_edges, 7),
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                9,
                2,
            )
            img_edges = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)

            img = cv2.bitwise_and(img_color, img_edges)

        elif self.type == "edges":
            img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)

        elif self.type == "rotate":
            rows, cols, _ = img.shape
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), frame.time * 45, 1)
            img = cv2.warpAffine(img, M, (cols, rows))

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# 🎥 Mixer (for multiple users)
def mixer_callback(frames: List[av.VideoFrame]) -> av.VideoFrame:
    buf_w, buf_h = 640, 480
    buffer = np.zeros((buf_h, buf_w, 3), dtype=np.uint8)

    n_inputs = len(frames)
    if n_inputs == 0:
        return av.VideoFrame.from_ndarray(buffer, format="bgr24")

    n_cols = math.ceil(math.sqrt(n_inputs))
    n_rows = math.ceil(n_inputs / n_cols)

    grid_w = buf_w // n_cols
    grid_h = buf_h // n_rows

    for i, frame in enumerate(frames):
        if frame is None:
            continue

        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]

        aspect = w / h
        new_w = min(grid_w, int(grid_h * aspect))
        new_h = min(grid_h, int(new_w / aspect))

        x = (i % n_cols) * grid_w + (grid_w - new_w) // 2
        y = (i // n_cols) * grid_h + (grid_h - new_h) // 2

        resized = cv2.resize(img, (new_w, new_h))
        buffer[y:y+new_h, x:x+new_w] = resized

    return av.VideoFrame.from_ndarray(buffer, format="bgr24")


# 🚀 Main App
def main():
    st.title("📹 Video Chat App")

    # Session state
    if "mix_track" not in st.session_state:
        st.session_state["mix_track"] = create_mix_track(
            kind="video", mixer_callback=mixer_callback, key="mix"
        )

    mix_track = st.session_state["mix_track"]

    ctx = webrtc_streamer(
        key="video",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio": True},
        source_video_track=mix_track,
        sendback_audio=False,
    )

    if ctx.input_video_track:
        process_track = create_process_track(
            input_track=ctx.input_video_track,
            processor_factory=OpenCVVideoProcessor,
        )

        mix_track.add_input_track(process_track)

        process_track.processor.type = st.radio(
            "Select Filter",
            ("noop", "cartoon", "edges", "rotate"),
        )


# Run app
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()