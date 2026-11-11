# BeyondMimic notes

Answers to the reading questions in the [week plan](plans/2026-11-09-week.md#beyondmimic-reading-questions). Paper: [arXiv 2508.08241](https://arxiv.org/abs/2508.08241). Code: `third_party/mjlab/src/mjlab/tasks/tracking/` (primary) and `third_party/whole_body_tracking/` (original).

Where to look in the code:

| Question | mjlab file |
|---|---|
| Observations, actions | `tasks/tracking/tracking_env_cfg.py`, `mdp/observations.py` |
| Reward, anchor body | `mdp/rewards.py`, `mdp/commands.py` |
| Adaptive sampling | `mdp/commands.py` |
| Domain randomization | `tracking_env_cfg.py` (events), `config/g1/env_cfgs.py` |
| PPO settings | `tasks/tracking/config/g1/` |
| PD gains, action scale | `asset_zoo/robots/unitree_g1/g1_constants.py` |

**How to read the citations.** Paper references are to arXiv v4 (13 Nov 2025): "M&M" is the *Materials and Methods* section, "S1" to "S4" are the supplementary sections, "Table S*n*" the supplementary tables. Code references are `file:line` relative to `third_party/mjlab/src/mjlab/` at the pinned v1.6.0 commit (`0fb8a68`), unless written as `wbt:` for `third_party/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/`. Checked on 2026-11-09.

## 1. Policy inputs and outputs ✅

**In: the reference's joint state, where the reference torso is relative to the robot's torso, and the robot's own proprioception. Out: 29 joint position setpoints for PD controllers.** The policy never sees the reference's body positions directly. All it gets from the reference is the joint angles and velocities at the current frame, plus a 9-D anchor-pose error.

Actor observation, paper (M&M, *Observation and Action*) against mjlab (`tasks/tracking/tracking_env_cfg.py:47-80`):

| Term | Paper | mjlab term | Dim | Training noise (mjlab) |
|---|---|---|---|---|
| Motion phase: reference joint pos + vel | $\psi = [\theta^{ref}, \dot\theta^{ref}]$, "solely as a progress cue" | `command` (`mdp/commands.py:126-128`) | 58 | none |
| Anchor position error | $e_{p,anchor}$ | `motion_anchor_pos_b` (`mdp/observations.py:18-28`) | 3 | ±0.25 m |
| Anchor orientation error | first two columns of $R^{des}R^\top$ (Rot6D) | `motion_anchor_ori_b` (`mdp/observations.py:31-41`) | 6 | ±0.05 |
| IMU twist | $\mathcal V_{imu}$ in the IMU frame | `base_lin_vel`, `base_ang_vel` (IMU sensors) | 3 + 3 | ±0.5, ±0.2 |
| Joint pos relative to default | $\theta - \theta^0$ | `joint_pos` with `biased=True` (encoder bias, see Q4) | 29 | ±0.01 rad |
| Joint velocities | $\dot\theta$ | `joint_vel` | 29 | ±0.5 rad/s |
| Last action | $a_{last}$ | `actions` | 29 | none |

That is 160 inputs, with no history stacking. The paper found stacking history *hurt* sim-to-real, and quaternion or axis-angle orientations did worse than Rot6D (M&M, *MDP Design and Parameters*, Fig. 8A). The paper gives no observation-noise ranges; the noise column is mjlab's own (`tracking_env_cfg.py:54-77`).

- **Critic (asymmetric).** It gets everything above without noise, plus every tracked body's position and Rot6D orientation relative to the robot's anchor (`tracking_env_cfg.py:82-107`, `mdp/observations.py:44-69`). With 14 tracked bodies that is 286 inputs. This matches S1, *Observation*.
- **No-State-Estimation variant.** S1 says that without reliable state estimation the *linear* parts may be dropped: the anchor position error and the linear IMU velocity. mjlab's `-No-State-Estimation` task does exactly that and keeps the anchor orientation (`config/g1/env_cfgs.py:74-84`, registered in `config/g1/__init__.py:15-21`). That leaves 154 actor inputs. The critic keeps everything.
- **Output.** $\theta^{sp} = \theta^0 + \alpha \odot a$ (M&M), with $\alpha_j = 0.25\,\tau_{max,j}/k_{p,j}$ (S1, *Heuristically Designed Parameters*). mjlab sets the base scale to 0.5 (`tracking_env_cfg.py:126-133`) and the G1 config overwrites it with exactly the paper's heuristic (`config/g1/env_cfgs.py:37`, computed in `asset_zoo/robots/unitree_g1/g1_constants.py:280-288`). The setpoints are deliberately not clipped to joint limits.
- **PD gains** follow S1: $k_p = I\omega^2$, $k_d = 2I\zeta\omega$ with $I$ the reflected rotor inertia (armature), $\omega = 10$ Hz and $\zeta = 2$. mjlab uses the same values (`g1_constants.py:112-123`).
- **Rates.** Physics runs at 200 Hz (`timestep=0.005`) with decimation 4, so the policy runs at 50 Hz (`tracking_env_cfg.py:306-311`). The paper also gives 50 Hz (S4, *Motion Tracking*).

## 2. Tracking reward and the anchor body ✅

**The reward is four Gaussian tracking scores (body position, orientation, linear and angular velocity, each averaged over the tracked bodies), two optional global anchor scores, and three penalties.** No term is motion-specific (M&M, *Rewards*; S1, *Reward*; Table S1).

$r(\bar e_s,\sigma_s) = \exp(-\bar e_s/\sigma_s^2)$, where $\bar e_s$ is the mean squared error over the target bodies.

| Term | Paper weight, σ | mjlab weight, σ (`tracking_env_cfg.py:209-251`) | Same? |
|---|---|---|---|
| Body position (anchor-relative) | 1.0, 0.3 m | 1.0, 0.3 | yes |
| Body orientation (anchor-relative) | 1.0, 0.4 rad | 1.0, 0.4 | yes |
| Body linear velocity (global) | 1.0, 1.0 m/s | 1.0, 1.0 | yes |
| Body angular velocity (global) | 1.0, 3.14 rad/s | 1.0, 3.14 | yes |
| Anchor position (global, "optional") | 0.5, 0.3 | 0.5, 0.3 (`motion_global_root_pos`) | yes, and on |
| Anchor orientation (global, "optional") | 0.5, 0.4 | 0.5, 0.4 (`motion_global_root_ori`) | yes, and on |
| Action rate $\lVert a_t - a_{t-1}\rVert^2$ | −0.1 | −0.1 (`envs/mdp/rewards.py:58-65`, squared) | yes |
| Joint limit (soft limits = 0.9 × hardware) | −10.0 | −10.0, soft factor 0.9 (`g1_constants.py:262`) | yes |
| Self-contact | **−0.1**, counts non-end-effector bodies with self-contact force > **1 N** | **−10.0**, counts the substeps (out of a 4-substep history) in which any self-contact in the pelvis subtree exceeds **10 N** (`mdp/rewards.py:116-135`, sensor at `config/g1/env_cfgs.py:24-33`) | **no** |

- **Self-contact is the one real difference.** The original `whole_body_tracking` matches the paper (−0.1, 1 N, end-effectors excluded: `wbt:tracking_env_cfg.py:238-250`). mjlab's term can reach 4 counts per step at weight −10, so a sustained self-collision costs far more than any tracking term can pay back. It also includes the hands and feet. This matters for fencing: a retargeted arm that clips the torso will be pushed away hard. [ADR 0002](adr/0002-saber-grip-and-mount.md) keeps the saber out of this sensor by making its geoms visual only.
- mjlab multiplies every reward by the step duration, 0.02 s (`managers/reward_manager.py:34-40`). The relative weights are unaffected.

**The anchor.** One body, `torso_link` for the G1 (`config/g1/env_cfgs.py:41`), follows the reference in world coordinates. Every other target body is tracked *relative to the robot's current anchor*. The reference is translated in xy under the robot, keeps the reference height, and is rotated by the yaw difference between the robot's anchor and the reference anchor (S1, *Anchor*; `mdp/commands.py:379-407`). Velocities are left global.

Why: pushes and the sim-to-real gap always cause some global drift. Tracking global body poses would make the policy fight that drift at the cost of the motion's style. Anchor-relative targets let the robot drift a bit and still do the right motion. The weak global anchor terms (0.5 each) and the anchor-error observation then pull it back gently (M&M, *Tracking Objective*, Fig. 7A).

The 14 tracked bodies (`config/g1/env_cfgs.py:42-57`) are the pelvis, hip-roll, knee and ankle-roll links of each leg, the torso, and the shoulder-roll, elbow and wrist-yaw links of each arm.

**Terminations** (S1, *Termination and Adaptive Sampling*; `tracking_env_cfg.py:257-279`, `mdp/terminations.py`):

| Condition | Paper | mjlab |
|---|---|---|
| Anchor height error | \|e<sub>z</sub>\| > 0.25 m | same (`bad_anchor_pos_z_only`) |
| End-effector height error (ankles, wrists) | \|e<sub>z</sub>\| > 0.25 m | same, on the anchor-relative targets (`config/g1/env_cfgs.py:64-69`) |
| Anchor orientation | ‖log(R<sup>des</sup>R<sup>⊤</sup>)‖ > 0.8 rad | difference in the z component of projected gravity > 0.8 (`mdp/terminations.py:37-53`). That checks tilt only, so a pure yaw error never terminates |
| Time-out | not stated in the paper | 10 s episodes (`tracking_env_cfg.py:312`) |

## 3. Adaptive sampling ✅

**Each episode starts at a point in the clip that the policy currently fails on, with a floor that keeps easy parts in the mix.** This is the paper's main tool for long clips whose difficulty is uneven (M&M, *Adaptive Sampling*; S1, *Termination and Adaptive Sampling*).

1. The clip is cut into one-second bins. In mjlab, `bin_count = T // 50 + 1` at 50 Hz (`mdp/commands.py:93`).
2. When an episode *terminates* (a time-out does not count), the bin it was in gets a failure (`mdp/commands.py:250-259`; `terminated` excludes time-outs, `managers/termination_manager.py:76-111`).
3. Per-bin failures are smoothed with an EMA, $\bar f \leftarrow 0.999\,\bar f + 0.001 f$, once per environment step (`mdp/commands.py:438-443`).
4. A uniform floor of $0.1/S$ is added to every bin, so the sampler drifts back to uniform once failures stop (`mdp/commands.py:262-264`).
5. The paper then convolves with a non-causal kernel $\rho^u$ ($\rho = 0.8$, $u \in \{0,1,2\}$), which raises the bins just *before* a failure, where the bad action was probably taken.
6. A bin is drawn from the normalized distribution, and the start frame is uniform within that bin (`mdp/commands.py:274-283`).
7. The robot is placed on the reference state at that frame, with pose noise (x, y ±0.05 m, z ±0.01 m, roll and pitch ±0.1 rad, yaw ±0.2 rad) and velocity noise equal to the push ranges. Both match S1 (`tracking_env_cfg.py:144-152`, applied in `mdp/commands.py:320-377`).

Where mjlab differs:

| | Paper | mjlab |
|---|---|---|
| Kernel width | K = 3 (u ∈ {0,1,2}) | `adaptive_kernel_size = 1` (`mdp/commands.py:616`), so the convolution does nothing. `whole_body_tracking` also defaults to 1 (`wbt:mdp/commands.py:368`). Neither code base uses the look-back kernel unless you set it |
| What is smoothed | a failure *rate* per bin | a raw failure *count* per bin per step, not divided by how often the bin was sampled (`mdp/commands.py:259`) |
| Joint perturbation at reset | not mentioned | joint positions get ±0.1 rad of noise (`tracking_env_cfg.py:153`). `wbt` does the same (`wbt:tracking_env_cfg.py:98`) |
| Play and evaluation | – | `sampling_mode="start"`, so it always starts at frame 0, with no reset noise and no pushes (`config/g1/env_cfgs.py:87-98`) |

λ = 0.8, α = 0.001 and the 0.1 floor all match the paper (`mdp/commands.py:617-619`). The ablation in Fig. 8B: without adaptive sampling, 3 of 4 motions still failed on hard segments (cartwheels) after 30k iterations, and an easy motion took 4k iterations instead of 2k.

## 4. Domain randomization ✅

**The paper randomizes only a compact set: ground contact, joint calibration offsets and the torso COM, plus random pushes.** The argument is that the actuators are modelled accurately (armature, manufacturer parameters, no system identification), so DR only has to cover what is truly uncertain. Heavier DR made behaviour conservative and harder to train (M&M, *Domain Randomization*; Intro). Values are from S1 and Table S2, and mjlab's are at `tracking_env_cfg.py:165-203`.

| Parameter | Paper | mjlab | Same? |
|---|---|---|---|
| Friction | static U(0.3, 1.6), dynamic U(0.3, 1.2), all bodies | one MuJoCo sliding friction U(0.3, 1.2), **foot geoms only**, shared across the foot geoms (`tracking_env_cfg.py:193-202`, geoms at `config/g1/env_cfgs.py:59-61`) | **no** |
| Restitution | U(0, 0.5) | not randomized | **no** |
| Default joint offset (calibration error) | U(−0.01, 0.01) rad, applied to actions *and* observations. The S1 text says ±0.1 rad for the ankles, but Table S2 lists only ±0.01 | `encoder_bias` U(−0.01, 0.01) on **all** joints, ankles included (`tracking_env_cfg.py:185-192`). It is added to the observed joint position (`entity/data.py:388-390`) and subtracted from the action target (`envs/mdp/actions/actions.py:222-223`), so it does hit both | yes, apart from the ankle ±0.1 in the text |
| Torso COM offset | x ±0.025, y ±0.05, z ±0.05 m | same, on `torso_link` (`tracking_env_cfg.py:172-184`, `config/g1/env_cfgs.py:62`) | yes |
| Pushes (set the root velocity) | every U(1, 3) s; lin (±0.5, ±0.5, ±0.2) m/s, ang (±0.52, ±0.52, ±0.78) rad/s | same (`tracking_env_cfg.py:30-37, 166-171`) | yes |
| When drawn | "at the start of each training episode unless noted" | `mode="startup"`: once per env for the whole run. The pushes are on an interval | **no** (and `wbt` also uses `startup`, `wbt:tracking_env_cfg.py:158-178`) |

mjlab also adds observation noise to the actor (Q1 table), which the paper does not describe.

**Why it matters for the real robot.** The calibration offset teaches the policy not to trust absolute joint readings to better than about 0.01 rad. Friction and pushes make it recover from slips and disturbances it never saw in the reference. The COM offset covers the battery and payload shifts the model gets wrong. The paper also measured what DR does *not* fix: wrong armature (overswing and self-collisions) and deployment latency (2 ms raised the velocity error, 5 ms caused one failure, 10 ms caused two failures in three trials; M&M, *MDP Design and Parameters*). Those have to be got right in the model and in the deployment code instead.

For us: the saber's mass is an extra randomization on top of these (`saber_mass`, [ADR 0002](adr/0002-saber-grip-and-mount.md) and [saber.md](saber.md)).

## 5. Why one recipe works, and where it fails ✅

**It works because nothing in the formulation knows which motion it is tracking.** Every motion is trained as its own policy, but with the same MDP and hyperparameters (M&M, *Scalable Human Motion Tracking via RL*). The paper attributes the generality to four choices:

1. **The reward is motion-agnostic:** uniform weights over the target bodies, task-space errors, σ chosen "to fit most motions", and only three regularizers (S1, *Reward*).
2. **Anchor-relative targets** tolerate drift, so a push does not turn into an unrecoverable error (Q2).
3. **The robot is modelled accurately.** Armature comes from rotor inertia × gear ratio², the PD gains are low-impedance (10 Hz, ζ = 2) and the action scale is tied to the torque limit. With that, DR can stay small and does not water down the objective (S1; Fig. 8A shows that zero armature caused overswing and self-collisions). In the PD ablation, 10 Hz tracked best globally. 25 Hz had slightly lower local error but made the gearboxes oscillate and grind (S2, *Extra Ablation on PD Gains*).
4. **Adaptive sampling** lets clips several minutes long converge despite a few hard segments (Q3).

The evidence is about 2.5 h of motion, all validated in simulation, and 30 clips (15 min) run on hardware (Results, *Diverse Skills*). The PPO hyperparameters in Table S5 are identical to mjlab's (`config/g1/rl_cfg.py:10-46`): MLPs of 512-256-128 with ELU, 24 steps per env, lr 1e-3 adaptive, clip 0.2, entropy 0.005, γ 0.99, λ 0.95, desired KL 0.01, 5 epochs, 4 mini-batches, 30,000 iterations.

**Where it fails.** The paper says all of its motions were validated in simulation and does not list motion types that the tracker cannot learn. What it does report:

- **Hard segments without adaptive sampling.** Cartwheels stayed unsolved after 30k iterations (Fig. 8B).
- **Deployment latency.** 5 to 10 ms of delay caused failures, and the paper chose not to randomize delay because that makes learning harder (M&M, *MDP Design and Parameters*).
- **Contact-rich ground motions** such as getting up from the floor break state estimation. For those the authors used LiDAR-inertial odometry or dropped the state-dependent observations, which is the No-State-Estimation variant (S4, *Motion Tracking*).
- **Hardware limits show through.** Walking ground-reaction forces peak more sharply than a human's because the G1 has no toe joint (Results, *Natural, Human-like Behaviors*).

*Our inference, not from the paper:* the tracker can only reproduce what is physically possible for a G1. Retargeting artifacts are reference errors the reward cannot fix: foot skating, floor penetration, self-penetration, joints past their limits. A reference whose torso or hand heights are off by more than 0.25 m will terminate every episode at the same spot, and the sampler will then pile onto that segment. Two things are out of scope for this MDP: contact with objects (there is no object or blade term in the reward) and anything that needs the robot to react to an opponent. In mjlab specifically, references where the sword arm crosses the body are at risk from the heavy self-contact penalty (Q2).

## 6. What guided diffusion adds (skim) ✅

**Tracking gives one policy per clip and no way to change the goal at run time. The diffusion stage turns many tracking policies into one model that can be *steered* at test time by a differentiable cost, without retraining** (M&M, *Versatile Humanoid Control via Guided Diffusion*; S3; S4).

1. **Distil into a latent action space.** A conditional VAE is trained with DAgger to imitate many tracking policies. The encoder sees only the reference part of the observation ($\psi$, $e_{anchor}$), producing a 32-D latent. The decoder maps that latent plus proprioception to joint setpoints (Table S6).
2. **Learn trajectories in that space.** A transformer diffusion model (about 19.8 M parameters) is trained on VAE rollouts. Each sample is a state–latent trajectory: 4 history steps and a 16-step horizon, which is 0.64 s at 25 Hz (Table S7, *Limitations*). The rollouts are perturbed with Ornstein–Uhlenbeck action noise so the model learns to recover from errors (S3, *Diffusion Dataset Collection*).
3. **Steer at test time.** Classifier guidance adds $-\nabla G(\tau)$ for a task cost $G$ at every denoising step (M&M, *Online Optimization via Guidance*). The paper's costs are joystick velocity, waypoint, an SDF obstacle barrier, and keyframe inpainting (S3, *Task Costs*). Costs can be summed to combine tasks.
4. **Deployment.** The model takes 20 denoising steps, about 20 ms, with TensorRT on an RTX 4060 Mobile, asynchronously. The light decoder runs synchronously on the CPU. The tracking policies for this stage are retrained at 25 Hz (S4, *Diffusion*).

Results: walking and running from a joystick (velocity error 12–14 % in simulation), waypoint navigation, obstacle avoidance, and transitions into cartwheels from keyframes. The latent version reached 95 % cartwheel success sim-to-sim, against 5 % for the version without a latent (M&M, *Latent Diffusion*). The paper's own limitations: the short 0.64 s horizon, sticking to the history, stumbles at the start and end of motions, and guidance that works for coarse goals but less well for fine-grained ones (*Limitations and Future Works*).

*Why it matters for us (inference):* Goal 2 (targeted hits) and Goal 4 (reactive fencing) need the goal to change at run time. A blade-tip-to-target distance is exactly the kind of differentiable cost guidance takes. But the paper's own caveat about fine-grained objectives is a warning for hitting a target to within a few centimetres. Also, the diffusion state as published has no blade in it. It would need the saber's tip in $s$, or forward kinematics on the predicted body poses.

## 7. Training time and GPU ✅

**The paper does not say how long one motion takes to train, or on what GPU.** It gives only the iteration budget: 30,000 maximum (Table S5). Fig. 8B also shows that an easy motion converges in about 2k iterations with adaptive sampling (4k without), and that hard ones need much longer. The only hardware it names is for deployment: tracking policies run on the robot's CPU with ONNX Runtime in under 1 ms per step, and diffusion runs on an RTX 4060 Mobile (S4). The training GPU is unverified.

**Our numbers** (RTX 5080 16 GB, WSL2, mjlab v1.6.0, 4096 envs × 24 steps = 98,304 transitions per iteration):

| Condition | s / iteration | 10k iterations | 30k (mjlab default, `config/g1/rl_cfg.py:45`) |
|---|---|---|---|
| Clean run | ~0.95 | ≈ 2.7 h | ≈ 8 h |
| Heavy CPU contention | 3.6–5.6, GPU utilisation 3 % | 10–16 h | 30–47 h |

- **Source of the clean figure:** `~/hcf-data/logs/train_dance1_subject2_p12v2_2026-11-09_14-56-51.log` (10,000-iteration run). Its first 100 iterations ran at about 0.8 s each (median per 50-iteration window: 0.82 s, then 0.78 s). Over iterations 0–247, collection (environment stepping) had a median of 0.83 s against 0.13 s for the PPO update, so about 85 % of each iteration is collection. Later in the same run, iterations 150–330 slowed to a median of 1.6–1.9 s while other work shared the machine. At the time of writing the GPU showed 83 % utilisation and 11 GB used.
- **Source of the contention figure:** the 3.6–5.6 s range and the 3 % GPU utilisation are our own observations from the run, not lines in that log. The p12v2 log never goes above 2.7 s. The earlier run `~/hcf-data/logs/train_p12_dance1.log` (411 iterations) does show contention: median 3.65 s per iteration (10th–90th percentile 2.1–8.3 s, maximum 16.5 s), with collection at 2.9 s.
- **What this means.** Under contention the PPO update barely changes while collection grows several-fold and the GPU sits idle. On this machine the loop is bound by CPU-side latency: Python and the kernel launches feeding MuJoCo Warp, not GPU throughput. Keep the CPU free during training; `nice` any other heavy job.

## 8. Reference motion format, and does GMR's output match? ✅

**Yes, it matches.** Checked against the pinned commits on 2026-11-08.

- Both stacks' `csv_to_npz.py` read a headerless CSV: `x,y,z, qx,qy,qz,qw, <29 joint angles>`. That is the root quaternion in **xyzw**, Unitree's LAFAN1 convention. They convert to wxyz internally, interpolate to 50 fps, run forward kinematics, and upload an NPZ (joint pos/vel plus per-body pose/velocity) to the W&B registry.
- GMR's `unitree_g1` model has the **identical 29-joint order** to the joint list in both `csv_to_npz.py` scripts. Checked programmatically: mjlab, whole_body_tracking and GMR all agree.
- GMR saves a pickle (`root_pos`, `root_rot` in xyzw, `dof_pos`, `fps`). GMR's own `scripts/batch_gmr_pkl_to_csv.py` writes exactly the CSV above. **No converter needed.**
- Gotcha: the NPZ is engine-specific (MuJoCo and PhysX order bodies differently). Convert each CSV separately for mjlab and for Isaac Lab ([ADR 0001](adr/0001-training-stack.md)).
