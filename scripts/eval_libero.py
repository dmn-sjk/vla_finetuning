"""Closed-loop LIBERO rollouts of a pi0 checkpoint; reports task success rate.

Protocol: 10 tasks x 50 trials from the suite's saved initial states, suite step cap.

    python scripts/eval_libero.py --verify-obs
    torchrun --nproc_per_node=8 scripts/eval_libero.py \
        --adapter <run>/adapter_best --out results/lora.json
"""

import argparse
import contextlib
import json
import os
import sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# renderer must be chosen before robosuite is imported; one GPU per rank
os.environ.setdefault("MUJOCO_GL", "egl")
_LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", str(_LOCAL_RANK))
os.environ.setdefault("EGL_DEVICE_ID", str(_LOCAL_RANK))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402
from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa: E402
from lerobot.policies.pi0.configuration_pi0 import PI0Config  # noqa: E402
from lerobot.utils.constants import ACTION  # noqa: E402

# LIBERO's own per-suite step budgets
SUITE_MAX_STEPS = {
    "libero_spatial": 280,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
# maps policy image keys to LIBERO cameras; dataset ports name the wrist camera differently
CAMERA_ALIASES = {
    "observation.images.image": "agentview_image",
    "observation.images.wrist_image": "robot0_eye_in_hand_image",
    "observation.images.image2": "robot0_eye_in_hand_image",
}

NUM_STEPS_WAIT = 10  # settle steps after reset, before the policy sees anything
DUMMY_ACTION = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)

# the dataset port rotates frames 180 degrees; check with --verify-obs
IMAGE_TRANSFORMS = {
    "none": lambda a: a,
    "flipud": lambda a: a[::-1],
    "fliplr": lambda a: a[:, ::-1],
    "rot180": lambda a: a[::-1, ::-1],
}


def quat2axisangle(quat):
    """robosuite xyzw quaternion to axis-angle, the 3 middle dims of the dataset state."""
    from robosuite.utils.transform_utils import quat2axisangle as _q2aa

    return _q2aa(np.asarray(quat))


def build_state(raw):
    return np.concatenate(
        [raw["robot0_eef_pos"], quat2axisangle(raw["robot0_eef_quat"]), raw["robot0_gripper_qpos"]]
    ).astype(np.float32)


def to_chw01(img_hwc_uint8):
    """Environment frame to the float32 CHW [0,1] tensor the dataset loader produces."""
    a = np.ascontiguousarray(img_hwc_uint8)
    return torch.from_numpy(a).permute(2, 0, 1).float().div_(255.0)


class Rank:
    """Minimal distributed helper: this script only ever needs rank/world and one gather."""

    def __init__(self):
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local = _LOCAL_RANK
        self.dist = self.world > 1
        if self.dist:
            # long timeout: ranks finish at very different times
            dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))

    def gather(self, obj):
        if not self.dist:
            return [obj]
        out = [None] * self.world
        dist.all_gather_object(out, obj)
        return out

    def close(self):
        if self.dist:
            dist.barrier()
            dist.destroy_process_group()


def load_policy(args, device, stats_meta):
    if args.from_pretrained_config:
        # inherit the checkpoint's structure (cameras, empty_cameras) from its config.json
        cfg = PI0Config.from_pretrained(args.policy)
        cfg.pretrained_path = args.policy
        cfg.device = device.type
        cfg.dtype = "bfloat16" if args.bf16 else "float32"
        # compile_model needs nvcc, which compute nodes lack
        cfg.compile_model = False
    else:
        cfg = PI0Config(
            pretrained_path=args.policy,
            device=device.type,
            dtype="bfloat16" if args.bf16 else "float32",
        )
    if args.n_action_steps is not None:
        cfg.n_action_steps = args.n_action_steps
    policy = make_policy(cfg, ds_meta=stats_meta)
    if args.adapter is not None:
        from peft import PeftModel

        policy = PeftModel.from_pretrained(policy, str(args.adapter))
    policy.to(device).eval()

    norm = {
        "stats": stats_meta.stats,
        "features": {**cfg.input_features, **cfg.output_features},
        "norm_map": cfg.normalization_mapping,
    }
    if args.checkpoint_stats:
        # third-party checkpoints ship the stats they were trained with
        pre_over = {"device_processor": {"device": device.type}}
        post_over = {}
    else:
        pre_over = {"device_processor": {"device": device.type}, "normalizer_processor": norm}
        post_over = {"unnormalizer_processor": norm}
    preprocessor, postprocessor = make_pre_post_processors(
        cfg,
        pretrained_path=args.policy,
        preprocessor_overrides=pre_over,
        # the unnormalizer needs the training stats too
        postprocessor_overrides=post_over,
    )
    _assert_unnormalizes(postprocessor, stats_meta, cfg, args)

    # skip empty_camera_* padding slots: pi0 fills them itself
    cams = []
    for key in cfg.image_features:
        if "empty_camera" in key:
            continue
        if key not in CAMERA_ALIASES:
            raise RuntimeError(
                f"policy expects image feature {key!r}, which has no LIBERO camera mapping. "
                f"Known: {sorted(CAMERA_ALIASES)}"
            )
        cams.append((key, CAMERA_ALIASES[key]))
    if not cams:
        raise RuntimeError(f"no usable camera inputs resolved from {list(cfg.image_features)}")
    print(f"camera wiring: " + ", ".join(f"{r} -> {k}" for k, r in cams))
    return policy, cfg, preprocessor, postprocessor, cams


def _assert_unnormalizes(postprocessor, stats_meta, cfg, args):
    """Check the postprocessor maps a zero action to the dataset action mean (MEAN_STD)."""
    import numpy as _np

    mean = _np.asarray(stats_meta.stats["action"]["mean"], dtype=_np.float32)
    dim = cfg.output_features[ACTION].shape[0]
    got = postprocessor(torch.zeros(1, cfg.chunk_size, dim)).float().cpu().numpy()[0, 0]
    if args.checkpoint_stats:
        # the checkpoint's stats need not match --dataset's; require only that it unnormalizes
        if not _np.isfinite(got).all() or _np.allclose(got, 0.0, atol=1e-6):
            raise RuntimeError(
                f"post-processor is not unnormalizing: unnormalize(0) returned {got.round(4)}. "
                "The checkpoint appears to ship no action stats; drop --checkpoint-stats."
            )
        print(f"unnormalize(0) = {got.round(4)} (checkpoint's own stats); "
              f"{args.dataset} action mean = {mean[:dim].round(4)}")
    elif not _np.allclose(got, mean[:dim], atol=1e-3):
        raise RuntimeError(
            "post-processor is not unnormalizing actions: unnormalize(0) returned "
            f"{got.round(4)}, expected the dataset action mean {mean[:dim].round(4)}. "
            "Rollouts would drive the robot with z-scores."
        )



def make_env(bddl_file, resolution):
    from libero.libero.envs import OffScreenRenderEnv

    # default OSC_POSE delta controller matches the dataset's action convention
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(0)
    return env


def verify_obs(args, device):
    """Compare env frames against real dataset frames under each candidate flip."""
    from libero.libero import benchmark, get_libero_path
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(0)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    init_states = torch.load(
        os.path.join(get_libero_path("init_states"), task.problem_folder, task.init_states_file),
        weights_only=False,
    )
    env = make_env(bddl, args.resolution)
    env.reset()
    raw = env.set_init_state(init_states[0])
    for _ in range(NUM_STEPS_WAIT):
        raw, _, _, _ = env.step(DUMMY_ACTION)

    ds = LeRobotDataset(args.dataset, episodes=[0])
    frame = ds[0]
    print(f"env task: {task.language!r}")
    print(f"dataset frame keys: {[k for k in frame if k.startswith('observation')]}")

    for cam, key in (
        ("agentview_image", "observation.images.image"),
        ("robot0_eye_in_hand_image", "observation.images.wrist_image"),
    ):
        ref = frame[key].float().numpy()  # CHW [0,1]
        print(f"\n{cam} vs {key}:")
        for name, fn in IMAGE_TRANSFORMS.items():
            cand = to_chw01(fn(raw[cam])).numpy()
            print(f"  {name:>7}: MSE={float(((cand - ref) ** 2).mean()):.5f}")
    env.close()
    print("\nPick the transform with the clearly lowest MSE on BOTH cameras.")


def rollout_batch(policy, preprocessor, postprocessor, envs, instructions, max_steps, transform,
                  device, autocast_ctx, cams):
    """Run one batch of episodes in lockstep. Returns per-episode (success, steps)."""
    n = len(envs)
    raws = []
    for env, init in envs:
        env.reset()
        raw = env.set_init_state(init)
        for _ in range(NUM_STEPS_WAIT):
            raw, _, _, _ = env.step(DUMMY_ACTION)
        raws.append(raw)

    policy.reset()
    done = [False] * n
    success = [False] * n
    steps = [max_steps] * n

    for t in range(max_steps):
        batch = {
            key: torch.stack([to_chw01(transform(r[raw])) for r in raws]) for key, raw in cams
        }
        batch["observation.state"] = torch.from_numpy(np.stack([build_state(r) for r in raws]))
        batch["task"] = list(instructions)
        with autocast_ctx():
            action = policy.select_action(preprocessor(batch))
        action = postprocessor(action).float().cpu().numpy()

        for i, (env, _) in enumerate(envs):
            if done[i]:
                continue  # finished episodes are frozen
            raw, _, term, _ = env.step(action[i])
            raws[i] = raw
            if env.check_success():
                success[i], done[i], steps[i] = True, True, t + 1
            elif term:
                done[i], steps[i] = True, t + 1
        if all(done):
            break
    return success, steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", type=Path, default=None, help="PEFT adapter dir; omit for a full checkpoint")
    p.add_argument("--policy", default="lerobot/pi0", help="base weights or a full fine-tuned dir")
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--dataset", default="lerobot/libero_spatial_image",
                   help="dataset supplying normalization stats (keep as the TRAINING set)")
    p.add_argument("--episodes-per-task", type=int, default=50, help="LIBERO convention: 50")
    p.add_argument("--task-ids", type=int, nargs="*", default=None)
    p.add_argument("--envs-per-rank", type=int, default=8)
    p.add_argument("--resolution", type=int, default=256, help="must match the dataset (256)")
    p.add_argument("--n-action-steps", type=int, default=5,
                   help="steps executed per 50-step chunk; openpi replans every 5")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--image-transform", default="rot180", choices=list(IMAGE_TRANSFORMS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-bf16", dest="bf16", action="store_false", default=True)
    p.add_argument("--from-pretrained-config", action="store_true",
                   help="inherit the checkpoint's config.json; for third-party checkpoints")
    p.add_argument("--checkpoint-stats", action="store_true",
                   help="use the checkpoint's own normalization stats; for third-party checkpoints")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--verify-obs", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(_LOCAL_RANK)

    if args.verify_obs:
        verify_obs(args, device)
        return

    R = Rank()
    torch.manual_seed(args.seed + 1000 * R.rank)

    from libero.libero import benchmark, get_libero_path

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task_ids = args.task_ids if args.task_ids is not None else list(range(len(suite.tasks)))
    max_steps = args.max_steps or SUITE_MAX_STEPS.get(args.suite, 500)
    transform = IMAGE_TRANSFORMS[args.image_transform]

    stats_meta = LeRobotDatasetMetadata(args.dataset)
    # an unseen prompt scores near zero, so warn on instructions absent from training
    train_tasks = {str(t[0]) for t in stats_meta.episodes.to_pandas()["tasks"]}
    for tid in task_ids:
        lang = suite.get_task(tid).language
        if lang not in train_tasks and R.rank == 0:
            print(f"[warn] task {tid} instruction {lang!r} is absent from {args.dataset}")

    policy, cfg, preprocessor, postprocessor, cams = load_policy(args, device, stats_meta)
    autocast_ctx = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if args.bf16 and device.type == "cuda"
        else contextlib.nullcontext
    )

    # deal (task, trial) round-robin so each rank gets a mix of tasks
    jobs = [(tid, ep) for tid in task_ids for ep in range(args.episodes_per_task)]
    mine = jobs[R.rank :: R.world]
    if R.rank == 0:
        print(f"{len(jobs)} episodes over {len(task_ids)} tasks, {R.world} rank(s), "
              f"{args.envs_per_rank} env(s)/rank, n_action_steps={cfg.n_action_steps}, "
              f"max_steps={max_steps}, transform={args.image_transform}", flush=True)

    # group by task so envs are reused across trials
    by_task = defaultdict(list)
    for tid, ep in mine:
        by_task[tid].append(ep)

    results = []
    for tid, eps in sorted(by_task.items()):
        task = suite.get_task(tid)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        init_states = torch.load(
            os.path.join(get_libero_path("init_states"), task.problem_folder, task.init_states_file),
            weights_only=False,
        )
        pool = [make_env(bddl, args.resolution) for _ in range(min(args.envs_per_rank, len(eps)))]
        try:
            for i in range(0, len(eps), len(pool)):
                chunk = eps[i : i + len(pool)]
                envs = [(pool[j], init_states[ep % len(init_states)]) for j, ep in enumerate(chunk)]
                succ, steps = rollout_batch(
                    policy, preprocessor, postprocessor, envs,
                    [task.language] * len(chunk), max_steps, transform, device, autocast_ctx,
                    cams,
                )
                for ep, s, st in zip(chunk, succ, steps):
                    results.append({"task_id": tid, "episode": ep, "success": bool(s), "steps": int(st)})
                nd = sum(r["success"] for r in results)
                print(f"[rank {R.rank}] task {tid} {i + len(chunk)}/{len(eps)} "
                      f"running {nd}/{len(results)}", flush=True)
        finally:
            for e in pool:
                e.close()

    gathered = [r for part in R.gather(results) for r in part]

    if R.rank == 0:
        per_task = defaultdict(list)
        for r in gathered:
            per_task[r["task_id"]].append(r["success"])
        overall = float(np.mean([r["success"] for r in gathered]))
        se = (overall * (1 - overall) / max(len(gathered), 1)) ** 0.5
        print("\n=== per-task success ===")
        for tid in sorted(per_task):
            v = per_task[tid]
            print(f"  task {tid}: {np.mean(v) * 100:5.1f}%  ({sum(v)}/{len(v)})  "
                  f"{suite.get_task(tid).language[:52]!r}")
        print(f"\nSUCCESS_RATE suite={args.suite} rate={overall * 100:.1f} "
              f"se={se * 100:.1f} n={len(gathered)} "
              f"n_action_steps={cfg.n_action_steps} checkpoint={args.adapter or args.policy}")
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            json.dump(
                {
                    "suite": args.suite,
                    "checkpoint": str(args.adapter or args.policy),
                    "policy": str(args.policy),
                    "stats_dataset": args.dataset,
                    "n_action_steps": cfg.n_action_steps,
                    "max_steps": max_steps,
                    "image_transform": args.image_transform,
                    "seed": args.seed,
                    "success_rate": overall,
                    "stderr": se,
                    "n_episodes": len(gathered),
                    "per_task": {str(k): float(np.mean(v)) for k, v in sorted(per_task.items())},
                    "episodes": sorted(gathered, key=lambda r: (r["task_id"], r["episode"])),
                },
                open(args.out, "w"),
                indent=2,
            )
            print(f"wrote {args.out}")
    R.close()


if __name__ == "__main__":
    main()
