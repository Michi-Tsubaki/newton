###########################################################################
# Example Robot Nextage Pick and Place
#
# Sets up a contact-rich bimanual pinch task for Nextage Open. The upstream
# NextageOpen URDF has arms only, so this example adds simple collision pads
# to the two wrist links and uses both arms as a parallel gripper.
#
# Command: uv run --extra example -m newton.examples robot_nextage_pick_place --world-count 1
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import JointTargetMode
from newton._src.utils.download_from_3rdparty import download_nextage_description
from newton.sensors import SensorContact


@wp.kernel
def interpolate_nextage_targets_kernel(
    q0: wp.array[wp.float32],
    q1: wp.array[wp.float32],
    t: float,
    coords_per_world: int,
    arm_coord_count: int,
    # output
    joint_target_pos: wp.array[wp.float32],
):
    world_idx = wp.tid()
    base = world_idx * coords_per_world
    for j in range(arm_coord_count):
        joint_target_pos[base + j] = q0[j] * (1.0 - t) + q1[j] * t


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.world_count = args.world_count
        self.viewer = viewer
        self.device = wp.get_device()
        self.use_cuda_graph = args.cuda_graph

        self.table_height = 0.84
        self.table_half_extents = (0.32, 0.32, 0.5 * self.table_height)
        self.cube_size = 0.06
        self.cube_pos = wp.vec3(0.285, 0.0, self.table_height + 0.5 * self.cube_size + 0.002)
        self.place_pos = wp.vec3(0.40, 0.0, self.cube_pos[2])

        robot = self.build_nextage()
        self.arm_coord_count = robot.joint_coord_count

        template = robot
        self.add_pick_place_scene(template)
        self.bodies_per_world = template.body_count
        self.coords_per_world = template.joint_coord_count
        self.object_body_local = template.body_label.index("object")

        builder = newton.ModelBuilder()
        builder.replicate(template, self.world_count, spacing=(1.2, 1.2, 0.0))
        builder.add_ground_plane()
        self.model = builder.finalize()

        self.pad_contact_sensor = SensorContact(
            self.model,
            sensing_obj_shapes=["*left_pad*", "*right_pad*"],
            counterpart_shapes="*object*",
            measure_total=False,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.contacts = self.model.contacts()
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_contacts=False,
            solver="newton",
            integrator="implicitfast",
            cone="elliptic",
            njmax=400,
            nconmax=400,
            iterations=30,
            ls_iterations=60,
            impratio=100.0,
        )

        self.viewer.set_model(self.model)
        self.viewer.picking_enabled = False
        if hasattr(self.viewer, "set_world_offsets"):
            self.viewer.set_world_offsets((1.2, 1.2, 0.0))
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(pos=wp.vec3(0.7, -1.5, 1.15), pitch=-15.0, yaw=130.0)

        self.object_max_z = [self.cube_pos[2]] * self.world_count if args.test else None
        self.setup_joint_space_waypoints()
        self.capture()

    def build_nextage(self) -> newton.ModelBuilder:
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.default_joint_cfg.target_ke = 700.0
        builder.default_joint_cfg.target_kd = 70.0
        builder.default_joint_cfg.armature = 0.02

        asset_path = download_nextage_description()
        builder.add_urdf(
            asset_path / "urdf/NextageOpen.urdf",
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.95), wp.quat_rpy(0.0,0.0,wp.radians(-90.0))), # offset
            floating=False,
            enable_self_collisions=False,
        )

        # Keep imported robot geometry visible, but exclude it from contact.
        # The task contact surface is the pair of simple wrist pads below.
        for shape_idx in range(builder.shape_count):
            builder.shape_flags[shape_idx] &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)

        init_q = np.array(
            [0.0, 0.0, 0.2, 0.4, -0.6, -1.2, 0.8, 0.0, 0.0, -0.4, -0.6, -1.2, -0.8, 0.0, 0.0],
            dtype=np.float32,
        )
        builder.joint_q = init_q.tolist()
        builder.joint_target_pos = init_q.tolist()

        for i in range(len(builder.joint_target_ke)):
            builder.joint_target_ke[i] = 700.0
            builder.joint_target_kd[i] = 70.0
            builder.joint_effort_limit[i] = 150.0
            builder.joint_target_mode[i] = int(JointTargetMode.POSITION)

        pad_cfg = newton.ModelBuilder.ShapeConfig(density=250.0, margin=0.0, mu=1.2)
        left_wrist = builder.body_label.index("NextageOpen/LARM_JOINT5_Link")
        right_wrist = builder.body_label.index("NextageOpen/RARM_JOINT5_Link")
        builder.add_shape_box(
            body=left_wrist,
            hx=0.025,
            hy=0.018,
            hz=0.045,
            cfg=pad_cfg,
            label="left_pad",
            color=(0.2, 0.6, 1.0),
        )
        builder.add_shape_box(
            body=right_wrist,
            hx=0.025,
            hy=0.018,
            hz=0.045,
            cfg=pad_cfg,
            label="right_pad",
            color=(1.0, 0.45, 0.2),
        )
        return builder

    def add_pick_place_scene(self, builder: newton.ModelBuilder):
        shape_cfg = newton.ModelBuilder.ShapeConfig(density=500.0, margin=0.0, mu=1.0)
        shape_cfg.ke = 5.0e4
        shape_cfg.kd = 5.0e2
        shape_cfg.kf = 1.0e3

        builder.add_shape_box(
            body=-1,
            hx=self.table_half_extents[0],
            hy=self.table_half_extents[1],
            hz=self.table_half_extents[2],
            xform=wp.transform(wp.vec3(0.34, 0.0, 0.5 * self.table_height), wp.quat_identity()),
            cfg=shape_cfg,
            label="table",
            color=(0.45, 0.45, 0.42),
        )

        object_body = builder.add_body(xform=wp.transform(self.cube_pos, wp.quat_identity()), label="object")
        builder.add_shape_box(
            body=object_body,
            hx=0.5 * self.cube_size,
            hy=0.5 * self.cube_size,
            hz=0.5 * self.cube_size,
            cfg=shape_cfg,
            label="object",
            color=(0.1, 0.75, 0.35),
        )

    def setup_joint_space_waypoints(self):
        q_home = np.array(
            [0.0, 0.0, 0.2, 0.4, -0.6, -1.2, 0.8, 0.0, 0.0, -0.4, -0.6, -1.2, -0.8, 0.0, 0.0],
            dtype=np.float32,
        )
        q_approach = q_home.copy()
        q_approach[4] = -0.2
        q_approach[10] = -0.2

        q_lower_open = q_home.copy()
        q_lower_open[4] = 0.0
        q_lower_open[10] = 0.0
        q_lower_open[3] -= 0.25
        q_lower_open[9] += 0.25

        q_pinch = q_lower_open.copy()
        q_pinch[3] -= 0.35
        q_pinch[9] += 0.35

        q_lift = q_pinch.copy()
        q_lift[4] = -0.2
        q_lift[10] = -0.2

        q_place = q_lift.copy()
        q_place[3] += 0.25
        q_place[9] -= 0.25

        q_release = q_place.copy()
        q_release[3] += 0.35
        q_release[9] -= 0.35

        self.waypoint_targets = [
            wp.array(q_home, dtype=wp.float32, device=self.device),
            wp.array(q_approach, dtype=wp.float32, device=self.device),
            wp.array(q_lower_open, dtype=wp.float32, device=self.device),
            wp.array(q_pinch, dtype=wp.float32, device=self.device),
            wp.array(q_lift, dtype=wp.float32, device=self.device),
            wp.array(q_place, dtype=wp.float32, device=self.device),
            wp.array(q_release, dtype=wp.float32, device=self.device),
            wp.array(q_home, dtype=wp.float32, device=self.device),
        ]
        self.waypoint_durations = [0.8, 1.0, 1.0, 1.2, 1.4, 1.0, 1.0, 1.2]
        self.current_waypoint = 0
        self.time_in_waypoint = 0.0

    def capture(self):
        self.graph = None
        if self.use_cuda_graph and wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def set_joint_targets(self):
        duration = self.waypoint_durations[self.current_waypoint]
        next_waypoint = (self.current_waypoint + 1) % len(self.waypoint_targets)
        self.time_in_waypoint += self.frame_dt
        t = min(self.time_in_waypoint / duration, 1.0)
        wp.launch(
            interpolate_nextage_targets_kernel,
            dim=self.world_count,
            inputs=[
                self.waypoint_targets[self.current_waypoint],
                self.waypoint_targets[next_waypoint],
                t,
                self.coords_per_world,
                self.arm_coord_count,
            ],
            outputs=[self.control.joint_target_pos],
            device=self.device,
        )

        if self.time_in_waypoint >= duration:
            self.current_waypoint = next_waypoint
            self.time_in_waypoint = 0.0

    def simulate(self):
        self.state_0.clear_forces()
        self.state_1.clear_forces()
        for _ in range(self.sim_substeps):
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.solver.update_contacts(self.contacts, self.state_0)

    def step(self):
        self.set_joint_targets()
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.pad_contact_sensor.update(self.state_0, self.contacts)
        self.sim_time += self.frame_dt

        if self.object_max_z is not None:
            body_q = self.state_0.body_q.numpy()
            for world_idx in range(self.world_count):
                object_body = world_idx * self.bodies_per_world + self.object_body_local
                self.object_max_z[world_idx] = max(self.object_max_z[world_idx], float(body_q[object_body][2]))

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        force_matrix = self.pad_contact_sensor.force_matrix.numpy()
        self.viewer.log_scalar("Pad Contact Force", float(np.abs(force_matrix).max()), smoothing=10)
        self.viewer.end_frame()

    def test_final(self):
        if self.object_max_z is None:
            return
        for world_idx, max_z in enumerate(self.object_max_z):
            lift = max_z - self.cube_pos[2]
            assert lift > 0.08, f"World {world_idx}: cube lift was only {lift:.3f} m"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(world_count=1)
        parser.set_defaults(num_frames=540)
        parser.add_argument("--cuda-graph", action="store_true", help="Capture simulation steps in a CUDA graph.")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
