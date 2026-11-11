// Deterministic start for motion_tracking_controller's MuJoCo sim-to-sim (docs/mtc.md).
//
// mujoco_sim_ros2 steps in real time from its first moment, but walking_controller becomes
// active only after its spawner, the ONNX load and the controller switch: 0.2 to 3.5 s of sim
// time depending on CPU load, while the unpowered G1 topples at ~0.7-1.0 s. This plugin keeps
// the whole robot (free joint and hinges) frozen in `hold_qpos` with zero velocity until the
// first physics step in which any actuator command or applied joint force is non-zero, i.e.
// the first step the activated controller drives. Sim time keeps running during the hold, so
// the controller manager, the state estimator and the activation proceed as upstream; only the
// robot's state is pinned. The release is decided inside the physics loop, not by a ROS
// message, so it does not depend on executor latency.
//
// The held robot must also *read* as a still, supported robot: the state estimator integrates
// the IMU from its activation, before the release. Resetting qpos/qvel after each step left the
// sensors computed from the collapsing unpowered robot (accelerometer ~0, i.e. free fall), and
// the estimator handed the policy a pelvis sinking at 0.6-1.5 m/s. So each held step also
// applies, as qfrc_applied, the inverse-dynamics force that makes qacc = 0 in the held state:
// the step's own forward dynamics, sensors included, then see a robot at rest on its feet
// (accelerometer +g, gyro 0, every velocity 0), whenever ros2_control reads them.
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <mujoco/mujoco.h>
#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/rclcpp.hpp>

#include "mujoco_sim_ros2/mujoco_physics_plugin.hpp"

namespace hcf_mtc_hold {

class HoldPlugin : public mujoco_sim_ros2::MujocoPhysicsPlugin {
public:
  void Configure(rclcpp::Node::SharedPtr &node, rclcpp::NodeOptions /*cm_node_option*/, mjModel *model,
                 mjData *data) override {
    logger_ = node->get_logger();
    if (!node->has_parameter("hold_qpos")) {
      node->declare_parameter("hold_qpos", std::vector<double>{});
    }
    qpos_ = node->get_parameter("hold_qpos").as_double_array();
    if (static_cast<int>(qpos_.size()) != model->nq) {
      // A wrong-length vector means the qpos was computed for another model: refuse to start.
      RCLCPP_FATAL(logger_, "hold_qpos has %zu values, the model has nq = %d", qpos_.size(), model->nq);
      throw std::runtime_error("hcf_mtc_hold: hold_qpos does not match the model");
    }
    Hold(model, data);
    mj_forward(model, data);
    RCLCPP_INFO(logger_, "hcf_mtc_hold: holding the robot in the start pose (pelvis z %.3f m)", qpos_[2]);
  }

  void Reset(mjModel *model, mjData *data) override {
    holding_ = true;
    Hold(model, data);
  }

  void PreUpdate(mjModel *model, mjData *data) override {
    if (!holding_) return;
    // Runs after MujocoRos2ControlPlugin's PreUpdate: qfrc_applied/ctrl hold this step's commands,
    // plus whatever of our previous hold force nobody overwrote (PostUpdate does not run before
    // the next PreUpdate here: a first version that cleared it there released at 0.004 s on its
    // own free-joint force). Remove our own values so only a real command releases.
    hold_force_.resize(model->nv, 0.0);
    for (int i = 0; i < model->nv; ++i) {
      if (data->qfrc_applied[i] == hold_force_[i]) data->qfrc_applied[i] = 0.0;
    }
    if (Commanded(model, data)) {
      holding_ = false;
      int dof = 0;
      for (int i = 0; i < model->nv; ++i) {
        if (std::abs(data->qfrc_applied[i]) > std::abs(data->qfrc_applied[dof])) dof = i;
      }
      RCLCPP_INFO(logger_, "hcf_mtc_hold: released at sim time %.4f s (first non-zero command; largest "
                  "qfrc_applied %.4g at dof %d)", data->time, model->nv ? data->qfrc_applied[dof] : 0.0, dof);
      return;
    }
    Hold(model, data);
    mju_zero(data->qacc, model->nv);
    mj_inverse(model, data);
    mju_copy(data->qfrc_applied, data->qfrc_inverse, model->nv);
    mju_copy(hold_force_.data(), data->qfrc_inverse, model->nv);
  }

  void PostUpdate(mjModel *model, mjData *data) override {
    if (holding_) Hold(model, data);
  }

private:
  void Hold(const mjModel *model, mjData *data) const {
    mju_copy(data->qpos, qpos_.data(), model->nq);
    mju_zero(data->qvel, model->nv);
    mju_zero(data->qacc_warmstart, model->nv);
  }

  static bool Commanded(const mjModel *model, const mjData *data) {
    const auto nonzero = [](double v) { return v != 0.0; };
    return std::any_of(data->ctrl, data->ctrl + model->nu, nonzero) ||
           std::any_of(data->qfrc_applied, data->qfrc_applied + model->nv, nonzero);
  }

  rclcpp::Logger logger_ = rclcpp::get_logger("hcf_mtc_hold");
  std::vector<double> qpos_;
  std::vector<double> hold_force_;  // the qfrc_applied this plugin wrote last step
  bool holding_ = true;
};

}  // namespace hcf_mtc_hold

PLUGINLIB_EXPORT_CLASS(hcf_mtc_hold::HoldPlugin, mujoco_sim_ros2::MujocoPhysicsPlugin)
