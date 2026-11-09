# Motions

Cleaned, versioned G1 reference motions: the output of Phase 1.4 and the input to Phase 1.5. These CSVs are the source of truth shared by both training stacks ([ADR 0001](../docs/adr/0001-training-stack.md)).

## Format

One row per frame, 30 fps unless the notes say otherwise, no header, 36 columns:

```
x, y, z,  qx, qy, qz, qw,  <29 joint angles in radians>
```

Root position is in metres, z up. The root quaternion is **xyzw**. Joint order is the Unitree G1 29-DoF order, as produced by GMR's `batch_gmr_pkl_to_csv.py`: left leg (6), right leg (6), waist (3), left arm (7), right arm (7).

## Naming

`<motion>_v<N>.csv`, e.g. `en_garde_v1.csv`. Motion names match `data/video_log.md`: `en_garde`, `advance`, `retreat`, `extension`, `lunge`, `advance_lunge`, `parry_<name>`, `cut_<target>`. Never overwrite a version: fix the data and save it as `v<N+1>`.

## Checking a motion

```bash
scripts/replay_csv.sh motions/en_garde_v1.csv   # kinematic replay video in outputs/replay/
```

## Provenance

| File | Source clip (`video_log.md` #) | Segment | Hand | Cleanup done | Notes |
|---|---|---|---|---|---|
| | | | | | |
