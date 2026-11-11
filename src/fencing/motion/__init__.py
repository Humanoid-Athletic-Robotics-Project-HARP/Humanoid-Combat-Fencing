"""fencing.motion: Phase 1.4e motion cleanup (plus segmentation and mirroring).

Raw retargeted G1 CSV (HCF_DATA/retarget/csv/<clip_id>.csv) -> cleaned, versioned
REPO/motions/<motion>_v<N>.csv. Modules, in pipeline order:

    io          read/write/validate the 36-column CSV and the JSON sidecar
    kinematics  batched MuJoCo FK on the training MJCF (feet, toes, heels, hands)
    contacts    per-frame foot contact with hysteresis
    normalize   canonical placement, left/right mirroring, handedness
    segment     find + label en_garde / advance / retreat / lunge / advance_lunge, holds
    fixes       smoothing, limits, speed clamp, time scaling, drift, ground, foot slide
    quality     metrics + pass/warn/fail
    publish     versioned files + provenance row in motions/README.md
    render      offscreen kinematic replay (PNG contact sheet / MP4)
    pipeline    the whole chain with defaults; cli wraps it as `fencing motion`
    synthetic   IK-built ground-truth footwork for tests and demos
"""
