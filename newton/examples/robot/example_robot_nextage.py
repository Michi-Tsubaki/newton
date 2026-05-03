###########################################################################
# Example Robot Nextage
#
# Shows how to set up a simulation of a Nextage Open robot from a local URDF
# file and applies a sinusoidal trajectory to the joint targets.
#
# Command: uv run --extra example -m newton.examples robot_nextage --world-count 4
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.utils.download_from_3rdparty import download_nextage_description
from newton import JointTargetMode


@wp.kernel
def update_joint_targets_kernel(
    base_targets: wp.array[wp.float32],
    amplitudes: wp.array[wp.float32],
    joint_limit_lower: wp.array[wp.float32],
    joint_limit_upper: wp.array[wp.float32],
    time: wp.float32,
    dofs_per_world: int,
    # output
    joint_target_pos: wp.array[wp.float32],
):
    dof = wp.tid()
    local_dof = dof % dofs_per_world
    world = dof / dofs_per_world
    phase = float(local_dof) * 0.47 + float(world) * 0.31
    target = base_targets[dof] + amplitudes[dof] * wp.sin(time + phase)
    joint_target_pos[dof] = wp.clamp(target, joint_limit_lower[dof], joint_limit_upper[dof])


class Example:
    def __init__(self, viewer, args):
        self.fps = 50
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.world_count = args.world_count
        self.viewer = viewer
        self.device = wp.get_device()

        nextage = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(nextage)

        nextage.default_joint_cfg.target_ke = 500.0
        nextage.default_joint_cfg.target_kd = 50.0
        nextage.default_joint_cfg.armature = 0.01

        asset_path = download_nextage_description()
        nextage.add_urdf(
            asset_path / "urdf/NextageOpen.urdf",
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.95), wp.quat_rpy(0.0,0.0,wp.radians(-90.0))), # offset
            floating=False,
            enable_self_collisions=False,
        )

        init_q = np.array(
            [
                0.0,
                0.0,
                0.2,
                0.4,
                -0.6,
                -1.2,
                0.8,
                0.0,
                0.0,
                -0.4,
                -0.6,
                -1.2,
                -0.8,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )
        nextage.joint_q = init_q.tolist()
        nextage.joint_target_pos = init_q.tolist()

        for i in range(len(nextage.joint_target_ke)):
            nextage.joint_target_ke[i] = 500.0
            nextage.joint_target_kd[i] = 50.0
            nextage.joint_target_mode[i] = int(JointTargetMode.POSITION)

        builder = newton.ModelBuilder()
        builder.replicate(nextage, self.world_count, spacing=(1.5, 1.5, 0.0))
        builder.add_ground_plane()

        self.model = builder.finalize()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = None

        self.dofs_per_world = nextage.joint_dof_count
        base_targets = np.tile(init_q, self.world_count)
        amplitudes = np.tile(
            np.array(
                [
                    0.2,
                    0.25,
                    0.15,
                    0.25,
                    0.25,
                    0.2,
                    0.25,
                    0.35,
                    0.35,
                    0.25,
                    0.25,
                    0.2,
                    0.25,
                    0.35,
                    0.35,
                ],
                dtype=np.float32,
            ),
            self.world_count,
        )
        self.base_targets = wp.array(base_targets, dtype=wp.float32, device=self.device)
        self.amplitudes = wp.array(amplitudes, dtype=wp.float32, device=self.device)

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            disable_contacts=True,
        )

        self.viewer.set_model(self.model)
        self.viewer.set_world_offsets((1.5, 1.5, 0.0))
        self.viewer.set_camera(
            pos=wp.vec3(0.0, -3.0, 1.2),
            pitch=0.0,
            yaw=90.0,
        )

        self.capture()

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)

            wp.launch(
                update_joint_targets_kernel,
                dim=self.model.joint_dof_count,
                inputs=[
                    self.base_targets,
                    self.amplitudes,
                    self.model.joint_limit_lower,
                    self.model.joint_limit_upper,
                    self.sim_time,
                    self.dofs_per_world,
                ],
                outputs=[self.control.joint_target_pos],
                device=self.device,
            )

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "nextage links stay above the ground",
            lambda q, qd: q[2] > 0.05,
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(world_count=4)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
