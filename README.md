
Humanoid-Combat-Fencing
Teaching a humanoid robot to fence.

This is a community project from the Humanoid Athletic Robotics Project (HARP). The goal is a Unitree G1 humanoid (29 DoF) that performs real fencing footwork and attacks: en garde, advance, retreat, extension and lunge. The robot learns these motions from videos of human fencers.

Fencing is a hard test for whole-body control. It combines a low, wide stance, fast explosive lunges, precise arm extension, and balance while recovering. If a humanoid can fence, it can do many other athletic motions.

Approach
fencing video ──► human motion ──► G1 motion ──► tracking policy ──► simulation ──► real G1
                   (GVHMR)          (GMR)        (BeyondMimic RL)   (Isaac Sim,     (human-
                                                                     MuJoCo)        supervised)
Video → human motion. GVHMR estimates 3D human pose (SMPL-X) from ordinary single-camera fencing video.
Human → robot motion. GMR retargets that motion onto the G1's joints and proportions, producing a reference trajectory.
Robot motion → policy. BeyondMimic (arXiv 2508.08241) trains a whole-body motion-tracking policy with reinforcement learning in simulation. It uses one fixed training recipe for every motion.
Simulation → hardware. Each policy is validated in Isaac Sim and in MuJoCo sim-to-sim before any supervised test on the real robot.
First steps
The current focus is getting a few basic fencing motions working end to end:

Build the fencing motion dataset. Source and curate fencing video, then convert it into reference trajectories for the G1, using GVHMR video-based pose estimation and GMR motion retargeting to map human motion onto the robot's joints and proportions.
Train and evaluate tracking policies. Train whole-body motion-tracking policies with BeyondMimic to reproduce fencing actions. We improve motion data quality (foot contacts, vertical drift, segment selection) rather than the training recipe, and evaluate how well each motion is reproduced in Isaac Sim and MuJoCo before human-supervised hardware testing.
We start slow and simple, and add motions in this order:

Motion	Why
En garde stance	Static balance in a fencing posture
Advance / retreat	Footwork and weight transfer
Extension	Precise arm motion while balanced
Lunge	Fast, explosive, whole-body motion; the main challenge
Status
Phase	Description	Status
0	Repository setup and pinned upstream code	✅ Done
1	GPU environment check (Isaac Sim 4.5 / Isaac Lab v2.1.0)	⏳ Next
2	Train one existing motion end to end to prove the pipeline	Planned
3	Fencing video → G1 motion	Planned
4	Train and evaluate fencing motions in simulation	Planned
5	Human-supervised hardware tests	Planned
Later	Reactive fencing against an opponent (guided diffusion)	Stretch goal
The full roadmap is in docs/roadmap.md, and every run is recorded in docs/log.md.

Repository layout
Path	Contents
docs/	Roadmap, method notes, run log
data/video_log.md	Links to source fencing videos (never the videos themselves)
motions/	Small retargeted G1 motion CSVs
training/	Run notes and W&B links (no checkpoints)
deploy/	Deployment notes only
safety/checklist.md	Checklist for human-led hardware tests
scripts/	Our own glue scripts
third_party/	Upstream code as pinned git submodules
CLAUDE.md	Ground rules and task list for AI-assisted work
Getting started
git clone https://github.com/Humanoid-Athletic-Robotics-Project-HARP/Humanoid-Combat-Fencing.git
cd Humanoid-Combat-Fencing
# Fetch only the upstream repos you need (GMR is about 1.2 GB):
scripts/setup_third_party.sh whole_body_tracking   # just one
scripts/setup_third_party.sh --all                   # or everything
Upstream code stays upstream. See third_party/README.md for pinned commits and licenses, and follow each project's own install instructions.

Training requires Linux, an NVIDIA GPU, Isaac Sim 4.5 / Isaac Lab v2.1.0, and a Weights & Biases organization for the motion registry.

Safety
Nothing in this repository sends commands to a real robot. Hardware tests are run by people only, with a gantry or harness, cleared space, and foam or padded weapons only. See safety/checklist.md.

Contributing
We welcome fencers, roboticists and ML people. See CONTRIBUTING.md and CODE_OF_CONDUCT.md. Fencers can help right away by suggesting clear, single-fencer videos for data/video_log.md.

License
Code and docs in this repository are Apache-2.0 (see LICENSE). Third-party projects keep their own licenses. GVHMR and the SMPL/SMPL-X body models are restricted to non-commercial research use.

Acknowledgements
This project builds on BeyondMimic (Hybrid Robotics), GMR (Yanjie Ze et al.), and GVHMR (Zhejiang University 3D Vision Group).

