# =========================
# camera_webrtc.py  (v1.1 – selectable deviceId)
# =========================
"""
Ultra-minimal **real-time camera preview** using `streamlit-webrtc`.
Now supports **manual deviceId selection** so you can switch from the built-in
laptop camera to a USB cam if the browser exposes it.

### How to use
1. In a browser console, run:
   ```javascript
   navigator.mediaDevices.enumerateDevices().then(ds => console.log(ds));
   ```
   → Note the `deviceId` of your USB camera (usually the second video device).
2. Paste that string into the **Device ID** field below and press *Apply*.

Run:
```bash
pip install streamlit streamlit-webrtc opencv-python
streamlit run camera_webrtc.py --server.port 8501
```
"""

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av  # PyAV

class IdentityTransformer(VideoTransformerBase):
    """Return frames unchanged – fastest path."""
    def transform(self, frame: av.VideoFrame):  # type: ignore
        return frame

def main():
    st.set_page_config(page_title="Fast Camera", layout="centered")
    st.title("🚀 Fast Camera Preview (WebRTC) – v1.1")

    st.markdown("""
    **Browser camera selection**
    1. Open DevTools → Console.
    2. Run <code>navigator.mediaDevices.enumerateDevices()</code>.
    3. Copy the <code>deviceId</code> of your USB camera and paste below.
    """)

    device_id = st.text_input("Device ID (leave blank for default camera)")
    st.info("Click **Apply** to switch camera.")
    if st.button("Apply"):
        # Force rerun with new constraints
        st.session_state.applied_device_id = device_id
        st.experimental_rerun()

    # Build constraints
    if "applied_device_id" not in st.session_state:
        st.session_state.applied_device_id = ""
    did = st.session_state.applied_device_id.strip()
    if did:
        constraints = {"video": {"deviceId": {"exact": did}}, "audio": False}
        st.success(f"Using deviceId: {did}")
    else:
        constraints = {"video": True, "audio": False}
        st.info("Using default camera (no deviceId specified)")

    webrtc_streamer(
        key="fastcam",
        video_transformer_factory=IdentityTransformer,
        media_stream_constraints=constraints,
    )

if __name__ == "__main__":
    main()