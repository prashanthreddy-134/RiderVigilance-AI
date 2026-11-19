import sys
import time
import numpy as np

# Ensure UTF-8 output encoding for Windows terminal
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from core.cv_pipeline import ComputerVisionPipeline

def test_pipeline_drowsiness_and_yawn_logic():
    print("==================================================")
    print("RUNNING VERIFICATION TEST FOR EYE & YAWN ALERTS")
    print("==================================================")

    pipeline = ComputerVisionPipeline(use_simulation=True)
    
    # 1. Normal State Test
    pipeline.sim_scenario = "normal"
    res1 = pipeline.process_frame()
    print(f"[TEST 1] Normal State: EAR={res1['avg_ear']:.3f}, MAR={res1['mar']:.3f}, State={res1['prediction']['state_name']}, Message='{res1['alert_message']}'")
    assert res1['prediction']['state_code'] == 0, f"Expected state 0, got {res1['prediction']['state_code']}"
    assert res1['alert_message'] == "", "Normal state should have no alert message."

    # 2. Continuous Eye Closure Test (>= 2.0s)
    pipeline.sim_scenario = "microsleep_drowsy"
    print("\n[TEST 2] Simulating Continuous Eye Closure (>= 2.0s)...")
    
    eye_alarm_triggered = False
    for i in range(25):
        res = pipeline.process_frame()
        time.sleep(0.1)
        dur = res['closure_duration']
        if res['alarm_sound_trigger']:
            eye_alarm_triggered = True

        if i % 6 == 0 or dur >= 2.0:
            print(f"  Frame {i+1:02d}: Avg EAR={res['avg_ear']:.3f} | EyeClosed={res['is_eye_closed']} | Duration={dur:.2f}s | AlarmTriggered={res['alarm_triggered']} | SoundTrigger={res['alarm_sound_trigger']} | Msg='{res['alert_message']}'")

    assert res['closure_duration'] >= 2.0, "Closure duration should be >= 2.0s"
    assert res['prediction']['state_code'] == 3, f"Expected State 3, got {res['prediction']['state_code']}"
    assert "DROWSINESS DETECTED" in res['alert_message'], f"Expected Drowsiness message, got '{res['alert_message']}'"
    assert eye_alarm_triggered, "Sound trigger should have fired once at 2.0s threshold!"

    # 3. Reset Eyes Test
    print("\n[TEST 3] Simulating Eyes Reopening...")
    pipeline.sim_scenario = "normal"
    for _ in range(3):
        res = pipeline.process_frame()
        time.sleep(0.1)
    
    print(f"  After Reopen: EyeClosed={res['is_eye_closed']} | Duration={res['closure_duration']}s | State={res['prediction']['state_name']}")
    assert res['closure_duration'] == 0.0, "Closure timer should reset to 0.0s when eyes reopen!"

    # 4. Temporal Yawn Test (MAR > 0.50 for >= 1.5s)
    pipeline.sim_scenario = "yawning"
    print("\n[TEST 4] Simulating Sustained Wide Mouth Opening (Yawn >= 1.5s)...")
    
    yawn_alarm_triggered = False
    for i in range(35):
        res = pipeline.process_frame()
        time.sleep(0.1)
        y_dur = res['yawn_duration']
        if res['alarm_sound_trigger']:
            yawn_alarm_triggered = True

        if i % 6 == 0 or res['yawn_confirmed']:
            print(f"  Frame {i+1:02d}: MAR={res['mar']:.3f} | MouthOpen={res['is_mouth_open']} | YawnDuration={y_dur:.2f}s | YawnConfirmed={res['yawn_confirmed']} | SoundTrigger={res['alarm_sound_trigger']} | Msg='{res['alert_message']}'")

    assert res['yawn_confirmed'], "Yawn should be confirmed after 1.5s sustained open mouth!"
    assert res['prediction']['state_code'] == 2, f"Expected State 2 (Yawn Detected), got {res['prediction']['state_code']}"
    assert "YAWN DETECTED" in res['alert_message'], f"Expected Yawn message, got '{res['alert_message']}'"
    assert yawn_alarm_triggered, "Sound trigger should have fired once when Yawn confirmed!"

    # 5. Mouth Closes Test
    print("\n[TEST 5] Simulating Mouth Closing...")
    pipeline.sim_scenario = "normal"
    for _ in range(3):
        res = pipeline.process_frame()
        time.sleep(0.1)
    
    print(f"  After Mouth Close: MouthOpen={res['is_mouth_open']} | YawnDuration={res['yawn_duration']}s | YawnConfirmed={res['yawn_confirmed']}")
    assert res['yawn_duration'] == 0.0, "Yawn timer should reset when mouth closes!"

    # 6. Mobile Phone Detection Test (YOLOv8)
    print("\n[TEST 6] Simulating Mobile Phone Usage Detection (YOLOv8)...")
    pipeline.sim_scenario = "mobile_use"
    res_mobile = pipeline.process_frame()
    print(f"  Mobile Detected: {res_mobile['mobile_detected']} | Confidence: {res_mobile['mobile_confidence']} | State: {res_mobile['prediction']['state_name']} | Msg: '{res_mobile['alert_message']}'")
    assert res_mobile['mobile_detected'], "Mobile should be detected in mobile_use scenario!"
    assert res_mobile['prediction']['state_code'] == 4, f"Expected State 4 (Mobile in Use), got {res_mobile['prediction']['state_code']}"
    assert res_mobile['alert_message'] == "⚠️ DO NOT USE MOBILE WHILE DRIVING", f"Expected mobile alert message, got '{res_mobile['alert_message']}'"
    assert res_mobile['alarm_triggered'], "Alarm should trigger when mobile is detected!"

    # 7. Repeated Short Eye Closures Test (Early Sleepiness 1-2s, 3+ times)
    print("\n[TEST 7] Simulating Repeated Short Eye Closures (1-2s, 3+ events)...")
    pipeline.sim_t = 0.0
    pipeline.sim_scenario = "repeated_closures"
    sleepiness_alarm_triggered = False
    for i in range(130):
        res_sleepy = pipeline.process_frame()
        time.sleep(0.045)
        if res_sleepy['alarm_sound_trigger']:
            sleepiness_alarm_triggered = True
        if res_sleepy['sleepy_confirmed']:
            break

    print(f"  Short Closures Count: {res_sleepy['short_closure_count']}/3 | SleepyConfirmed: {res_sleepy['sleepy_confirmed']} | State: {res_sleepy['prediction']['state_name']} | Msg: '{res_sleepy['alert_message']}'")
    assert res_sleepy['sleepy_confirmed'], "Sleepiness should be confirmed after 3 short closures (1-2s)!"
    assert res_sleepy['prediction']['state_code'] == 5, f"Expected State 5 (Feeling Sleepy), got {res_sleepy['prediction']['state_code']}"
    assert "YOU ARE FEELING SLEEPY" in res_sleepy['alert_message'], f"Expected Feeling Sleepy message, got '{res_sleepy['alert_message']}'"
    assert sleepiness_alarm_triggered, "Alarm sound trigger should have fired when sleepiness threshold was reached!"

    print("\n==================================================")
    print("ALL EYE, YAWN, MOBILE & REPEATED CLOSURE TESTS PASSED!")
    print("==================================================")

if __name__ == "__main__":
    test_pipeline_drowsiness_and_yawn_logic()
