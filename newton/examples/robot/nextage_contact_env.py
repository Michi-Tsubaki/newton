# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton import JointTargetMode
from newton.sensors import SensorContact

try:  # Optional dependency: training/eval scripts use torch, the env itself does not require it.
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover - exercised when torch is unavailable
    torch = None
    nn = None

NEXTAGE_DESCRIPTION = {
    "url": (
        "https://raw.githubusercontent.com/iory/scikit-robot-models/"
        "35f450dde137629b641206be7cee5f262976b07d/nextage_description.tar.gz"
    ),
    "sha256": "df4eb9debfa0eb60d1ed14b6bd3c0a14bddb31d4f754ea766c2337671dd7e003",
    "root_dir": "nextage_description",
}


def _default_cache_dir() -> Path:
    return Path(__file__).resolve().parents[3] / ".cache" / "thirdparty_assets"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract_tar(tar: tarfile.TarFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in tar.getmembers():
        if member.issym() or member.islnk() or not (member.isdir() or member.isfile()):
            raise RuntimeError(f"Refusing to extract unsupported archive member: {member.name}")
        member_path = (destination / member.name).resolve()
        if os.path.commonpath([destination, member_path]) != str(destination):
            raise RuntimeError(f"Refusing to extract unsafe archive member: {member.name}")
    tar.extractall(destination)


def download_nextage_description(
    cache_dir: str | os.PathLike[str] | None = None,
    force_refresh: bool = False,
) -> Path:
    """Download and extract the Nextage description package.

    Args:
        cache_dir: Directory to cache downloads. Defaults to the repository cache directory.
        force_refresh: If True, redownload and re-extract the archive.

    Returns:
        Path to the extracted `nextage_description` directory.
    """
    cache_path = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
    final_dir = cache_path / f"nextage_description_{NEXTAGE_DESCRIPTION['sha256'][:8]}" / NEXTAGE_DESCRIPTION["root_dir"]
    marker = final_dir / ".newton_thirdparty_asset"

    if final_dir.exists() and marker.exists() and not force_refresh:
        return final_dir

    package_dir = final_dir.parent
    cache_path.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="nextage_description_", dir=cache_path))
    archive_path = temp_dir / "nextage_description.tar.gz"

    try:
        urllib.request.urlretrieve(NEXTAGE_DESCRIPTION["url"], archive_path)
        actual_sha256 = _sha256(archive_path)
        if actual_sha256 != NEXTAGE_DESCRIPTION["sha256"]:
            raise RuntimeError(
                "Checksum mismatch for nextage_description: "
                f"expected {NEXTAGE_DESCRIPTION['sha256']}, got {actual_sha256}"
            )

        extract_dir = temp_dir / "extract"
        extract_dir.mkdir()
        with tarfile.open(archive_path, "r:gz") as tar:
            _safe_extract_tar(tar, extract_dir)

        extracted_root = extract_dir / NEXTAGE_DESCRIPTION["root_dir"]
        if not extracted_root.exists():
            raise RuntimeError("Archive nextage_description does not contain nextage_description/")

        if package_dir.exists():
            shutil.rmtree(package_dir)
        package_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(extracted_root), str(final_dir))
        marker.write_text(f"{NEXTAGE_DESCRIPTION['url']}\n{NEXTAGE_DESCRIPTION['sha256']}\n", encoding="utf-8")
        return final_dir
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


@dataclass(frozen=True)
class PPOConfig:
    obs_dim: int
    action_dim: int
    hidden_dims: tuple[int, int, int] = (256, 256, 128)


if torch is not None and nn is not None:

    class PPOActorCritic(nn.Module):
        def __init__(self, config: PPOConfig):
            super().__init__()
            h1, h2, h3 = config.hidden_dims
            self.policy = nn.Sequential(
                nn.Linear(config.obs_dim, h1),
                nn.Tanh(),
                nn.Linear(h1, h2),
                nn.Tanh(),
                nn.Linear(h2, h3),
                nn.Tanh(),
                nn.Linear(h3, config.action_dim),
            )
            self.value = nn.Sequential(
                nn.Linear(config.obs_dim, h1),
                nn.Tanh(),
                nn.Linear(h1, h2),
                nn.Tanh(),
                nn.Linear(h2, h3),
                nn.Tanh(),
                nn.Linear(h3, 1),
            )
            self.log_std = nn.Parameter(torch.zeros(config.action_dim))

        def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            mean = self.policy(obs)
            value = self.value(obs).squeeze(-1)
            return mean, value

        def act(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            mean, value = self.forward(obs)
            std = torch.exp(self.log_std).expand_as(mean)
            dist = torch.distributions.Normal(mean, std)
            if deterministic:
                raw_action = mean
            else:
                raw_action = dist.rsample()
            action = torch.tanh(raw_action)
            logprob = dist.log_prob(raw_action) - torch.log(1.0 - action.square() + 1.0e-6)
            return action, logprob.sum(dim=-1), value

        def evaluate_actions(
            self, obs: torch.Tensor, actions: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            mean, value = self.forward(obs)
            std = torch.exp(self.log_std).expand_as(mean)
            dist = torch.distributions.Normal(mean, std)
            clipped = actions.clamp(-0.999, 0.999)
            raw_action = torch.atanh(clipped)
            logprob = dist.log_prob(raw_action) - torch.log(1.0 - clipped.square() + 1.0e-6)
            entropy = dist.entropy().sum(dim=-1)
            return logprob.sum(dim=-1), entropy, value
else:  # pragma: no cover - exercised only when torch is unavailable
    PPOActorCritic = None


class NextageLiftPlaceEnv:
    """Vectorized Nextage lift-place task built on Newton."""

    def __init__(
        self,
        num_envs: int = 64,
        seed: int = 1,
        headless: bool = True,
        viewer=None,
        use_cuda_graph: bool = True,
    ):
        self.num_envs = num_envs
        self.viewer = viewer
        self.headless = headless
        self.device = wp.get_device()
        self.rng = np.random.default_rng(seed)
        self.use_cuda_graph = use_cuda_graph and self.device.is_cuda

        self.fps = 50
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.episode_steps = 160
        self.table_height = 0.84
        self.table_half_extents = (0.32, 0.32, 0.5 * self.table_height)
        self.cube_size = 0.06
        self.cube_pos = wp.vec3(0.285, 0.0, self.table_height + 0.5 * self.cube_size + 0.002)
        self.place_pos = wp.vec3(0.40, 0.0, self.cube_pos[2])
        self.action_scale = 0.04
        self.lift_height = self.cube_pos[2] + 0.08
        self.success_xy_threshold = 0.045
        self.drop_height = self.table_height + 0.01

        template = self._build_template()
        self.arm_dof_count = template.joint_dof_count
        self.template_body_count = template.body_count
        self.template_shape_count = template.shape_count
        self.object_body_local = template.body_label.index("object")

        builder = newton.ModelBuilder()
        builder.replicate(template, self.num_envs, spacing=(1.2, 1.2, 0.0))
        builder.add_ground_plane()
        self.model = builder.finalize()

        self.bodies_per_world = self.template_body_count
        self.shapes_per_world = self.template_shape_count
        self.dofs_per_world = self.model.joint_dof_count // self.num_envs
        self.action_dim = self.dofs_per_world
        self.obs_dim = self.action_dim + 6

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        self.pad_sensor = SensorContact(
            self.model,
            sensing_obj_shapes=["*left_pad*", "*right_pad*"],
            counterpart_shapes="*object*",
            measure_total=False,
        )

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

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self._initial_joint_q = wp.clone(self.model.joint_q)
        self._initial_joint_qd = wp.clone(self.model.joint_qd)

        self.joint_limit_lower = np.asarray(self.model.joint_limit_lower.numpy(), dtype=np.float32)
        self.joint_limit_upper = np.asarray(self.model.joint_limit_upper.numpy(), dtype=np.float32)
        self._joint_target_np = np.asarray(self.model.joint_q.numpy(), dtype=np.float32).copy()
        self._joint_target_wp = wp.array(self._joint_target_np, dtype=wp.float32, device=self.device)
        self.control.joint_target_pos = self._joint_target_wp

        self._step_count = 0
        self._graph = None
        self._set_viewer()
        self.reset_all()
        self.capture()

    def _build_template(self) -> newton.ModelBuilder:
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.default_joint_cfg.target_ke = 700.0
        builder.default_joint_cfg.target_kd = 70.0
        builder.default_joint_cfg.armature = 0.02

        asset_path = download_nextage_description()
        builder.add_urdf(
            asset_path / "urdf" / "NextageOpen.urdf",
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.95), wp.quat_rpy(0.0, 0.0, wp.radians(-90.0))),
            floating=False,
            enable_self_collisions=False,
        )

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
        return builder

    def _set_viewer(self) -> None:
        if self.viewer is None:
            return
        self.viewer.set_model(self.model)
        self.viewer.picking_enabled = False
        if hasattr(self.viewer, "set_world_offsets"):
            self.viewer.set_world_offsets((1.2, 1.2, 0.0))
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(pos=wp.vec3(0.7, -1.5, 1.15), pitch=-15.0, yaw=130.0)

    def capture(self) -> None:
        self._graph = None
        if self.use_cuda_graph and self.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self._simulate()
            self._graph = capture.graph

    def reset_all(self) -> np.ndarray:
        wp.copy(self.state_0.joint_q, self._initial_joint_q)
        wp.copy(self.state_0.joint_qd, self._initial_joint_qd)
        wp.copy(self.state_1.joint_q, self._initial_joint_q)
        wp.copy(self.state_1.joint_qd, self._initial_joint_qd)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.state_1.joint_q, self.state_1.joint_qd, self.state_1)

        self._joint_target_np[...] = np.asarray(self.model.joint_q.numpy(), dtype=np.float32)
        wp.copy(self._joint_target_wp, wp.array(self._joint_target_np, dtype=wp.float32, device=self.device))
        self._step_count = 0
        self._refresh_contacts()
        return self.observe()

    def _simulate(self) -> None:
        self.state_0.clear_forces()
        self.state_1.clear_forces()
        for _ in range(self.sim_substeps):
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.solver.update_contacts(self.contacts, self.state_0)

    def _refresh_contacts(self) -> None:
        self.model.collide(self.state_0, self.contacts)
        self.solver.update_contacts(self.contacts, self.state_0)
        self.pad_sensor.update(self.state_0, self.contacts)

    def step(self, actions) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        actions = np.asarray(actions, dtype=np.float32)
        actions = np.clip(actions, -1.0, 1.0)
        self._joint_target_np += actions * self.action_scale
        np.clip(self._joint_target_np, self.joint_limit_lower, self.joint_limit_upper, out=self._joint_target_np)
        wp.copy(self._joint_target_wp, wp.array(self._joint_target_np, dtype=wp.float32, device=self.device))

        if self._graph is not None:
            wp.capture_launch(self._graph)
        else:
            self._simulate()

        self.pad_sensor.update(self.state_0, self.contacts)
        self._step_count += 1

        obs = self.observe()
        reward, metrics = self._compute_reward(actions)
        done = self._compute_done(metrics)
        info = {
            "success": metrics["success"],
            "lift_height": metrics["lift_height"],
            "goal_dist": metrics["goal_dist"],
            "contact_force": metrics["contact_force"],
            "object_z": metrics["object_z"],
        }
        return obs, reward, done, info

    def observe(self) -> np.ndarray:
        joint_q = np.asarray(self.state_0.joint_q.numpy(), dtype=np.float32).reshape(self.num_envs, self.dofs_per_world)
        force_matrix = np.asarray(self.pad_sensor.force_matrix.numpy(), dtype=np.float32).reshape(self.num_envs, 2, 3)
        return np.concatenate([force_matrix[:, 0, :], force_matrix[:, 1, :], joint_q], axis=1)

    def _compute_reward(self, actions) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        body_q = np.asarray(self.state_0.body_q.numpy(), dtype=np.float32).reshape(self.num_envs, self.bodies_per_world, -1)
        pad_tf = np.asarray(self.pad_sensor.sensing_obj_transforms.numpy(), dtype=np.float32).reshape(self.num_envs, 2, -1)
        force_matrix = np.asarray(self.pad_sensor.force_matrix.numpy(), dtype=np.float32).reshape(self.num_envs, 2, 3)

        object_pos = body_q[:, self.object_body_local, :3]
        left_pad_pos = pad_tf[:, 0, :3]
        right_pad_pos = pad_tf[:, 1, :3]
        left_force = np.linalg.norm(force_matrix[:, 0, :], axis=1)
        right_force = np.linalg.norm(force_matrix[:, 1, :], axis=1)

        dist_left = np.linalg.norm(left_pad_pos - object_pos, axis=1)
        dist_right = np.linalg.norm(right_pad_pos - object_pos, axis=1)
        reach_reward = np.exp(-8.0 * dist_left) + np.exp(-8.0 * dist_right)

        lift_height = np.maximum(object_pos[:, 2] - self.cube_pos[2], 0.0)
        lift_reward = np.clip(lift_height / 0.08, 0.0, 1.0)

        goal_dist = np.linalg.norm(object_pos[:, :2] - np.array([self.place_pos[0], self.place_pos[1]], dtype=np.float32), axis=1)
        place_gate = np.clip((lift_height - 0.04) / 0.04, 0.0, 1.0)
        place_reward = np.exp(-10.0 * goal_dist) * place_gate

        contact_force = np.minimum(left_force, 15.0) + np.minimum(right_force, 15.0)
        contact_reward = np.minimum(contact_force / 12.0, 1.0)

        release_bonus = np.where((goal_dist < self.success_xy_threshold) & (lift_height > 0.02), 1.0, 0.0)
        action_penalty = np.mean(np.square(actions), axis=1)
        drop_penalty = np.where(object_pos[:, 2] < self.drop_height, 1.0, 0.0)

        reward = (
            1.2 * reach_reward
            + 0.6 * contact_reward
            + 1.5 * lift_reward
            + 2.5 * place_reward
            + 0.8 * release_bonus
            - 0.02 * action_penalty
            - 1.5 * drop_penalty
        )
        success = np.where((goal_dist < self.success_xy_threshold) & (lift_height > 0.03) & (contact_force < 2.0), 1.0, 0.0)

        metrics = {
            "lift_height": lift_height.astype(np.float32),
            "goal_dist": goal_dist.astype(np.float32),
            "contact_force": contact_force.astype(np.float32),
            "success": success.astype(np.float32),
            "object_z": object_pos[:, 2].astype(np.float32),
        }
        return reward.astype(np.float32), metrics

    def _compute_done(self, metrics: dict[str, np.ndarray]) -> np.ndarray:
        horizon_done = np.full((self.num_envs,), self._step_count >= self.episode_steps, dtype=bool)
        drop_done = metrics["object_z"] < self.drop_height
        return horizon_done | drop_done

    def render(self) -> None:
        if self.viewer is None:
            return
        self.viewer.begin_frame(self._step_count * self.frame_dt)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


__all__ = [
    "NextageLiftPlaceEnv",
    "PPOActorCritic",
    "PPOConfig",
    "download_nextage_description",
]
