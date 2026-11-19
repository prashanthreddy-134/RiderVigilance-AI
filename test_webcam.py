import cv2

def test_cameras():
    print("Testing OpenCV VideoCapture indices...")
    for idx in range(5):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"[SUCCESS] Camera index {idx} opened cleanly! Frame shape: {frame.shape}")
                cap.release()
                return idx
            else:
                print(f"[WARNING] Camera index {idx} opened but failed reading frame.")
            cap.release()
        else:
            print(f"[INFO] Camera index {idx} could not be opened.")
    return None

if __name__ == "__main__":
    found_idx = test_cameras()
    if found_idx is None:
        print("[RESULT] No active camera index found.")
    else:
        print(f"[RESULT] Active camera found at index {found_idx}")
