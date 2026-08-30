"""Why does a checkpoint with a good validation loss score 0% in closed loop?

Three independent checks, each isolating one link of the chain, so the answer is a
measurement rather than a guess:

  1. OPEN LOOP -- feed the policy *dataset* observations and compare its predicted action
     chunk against the recorded expert actions. This exercises the policy and the processor
     pipeline with zero simulator involvement. If this is bad, the checkpoint or the
     preprocessing is at fault and the env is irrelevant.

  2. STATE MATCH -- reset the env, build our 8-D state from the raw observation, then find
     the nearest state among all dataset frames of the same task. The dataset's episodes are
     demonstrations on the same saved initial states, so a correct construction must land
     very close to some real frame. A large minimum distance means our state vector is wrong
     (wrong ordering, wrong sign, wrong units) no matter how plausible it looks.

  3. IMAGE MATCH -- having found the nearest dataset frame, compare its images to ours under
     each candidate transform. This is --verify-obs but anchored to a frame we know
     corresponds, instead of an arbitrary one.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("MUJOCO_GL", "egl")

import train_pi0 as T  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata  # noqa: E402
from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa: E402
from lerobot.policies.pi0.configuration_pi0 import PI0Config  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_libero import (  # noqa: E402
    DUMMY_ACTION, IMAGE_TRANSFORMS, NUM_STEPS_WAIT, build_state, make_env, to_chw01,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", type=Path, default=None)
    p.add_argument("--policy", default="lerobot/pi0")
    p.add_argument("--dataset", default="lerobot/libero_spatial_image")
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--frames", type=int, default=16, help="frames for the open-loop check")
    p.add_argument("--resolution", type=int, default=256)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = LeRobotDatasetMetadata(args.dataset)
    cfg = PI0Config(pretrained_path=args.policy, device=device.type, dtype="bfloat16")
    policy = make_policy(cfg, ds_meta=meta)
    if args.adapter is not None:
        from peft import PeftModel

        policy = PeftModel.from_pretrained(policy, str(args.adapter))
    policy.to(device).eval()
    norm = {
        "stats": meta.stats,
        "features": {**cfg.input_features, **cfg.output_features},
        "norm_map": cfg.normalization_mapping,
    }
    preprocessor, postprocessor = make_pre_post_processors(
        cfg, pretrained_path=args.policy,
        preprocessor_overrides={"device_processor": {"device": device.type}, "normalizer_processor": norm},
        postprocessor_overrides={"unnormalizer_processor": norm},
    )

    # ---------------- 1. open loop against recorded expert actions ----------------
    print("=" * 72)
    print("1. OPEN LOOP: predicted action chunk vs recorded expert actions (val episodes)")
    print("=" * 72)
    from types import SimpleNamespace

    _, val, n_tr, n_va = T.build_datasets(
        SimpleNamespace(dataset=args.dataset, val_episodes=40), meta, cfg
    )
    print(f"{n_tr} train / {n_va} val episodes ({len(val)} val frames)")
    idx = np.linspace(0, len(val) - 1, args.frames).astype(int)

    preds, gts = [], []
    for i in idx:
        f = val[int(i)]
        batch = {
            "observation.images.image": f["observation.images.image"].unsqueeze(0),
            "observation.images.wrist_image": f["observation.images.wrist_image"].unsqueeze(0),
            "observation.state": f["observation.state"].unsqueeze(0),
            "task": [f["task"]],
        }
        with torch.autocast("cuda", dtype=torch.bfloat16):
            chunk = policy.predict_action_chunk(preprocessor(batch))
        preds.append(postprocessor(chunk).float().cpu().numpy()[0])
        gts.append(f["action"].numpy())
    P, G = np.stack(preds), np.stack(gts)
    print(f"chunks {P.shape} vs ground truth {G.shape}")
    mae = np.abs(P - G).mean(axis=(0, 1))
    naive = np.abs(G - G.mean(axis=(0, 1), keepdims=True)).mean(axis=(0, 1))
    print("           " + "".join(f"{d:>9}" for d in ["dx", "dy", "dz", "rx", "ry", "rz", "grip"]))
    print("MAE      : " + "".join(f"{v:9.4f}" for v in mae))
    print("MAE(mean): " + "".join(f"{v:9.4f}" for v in naive), " <- predicting the dataset mean")
    print(f"\nfirst predicted step : {P[0, 0].round(3)}")
    print(f"first expert step    : {G[0, 0].round(3)}")
    print(f"\n>>> policy beats the mean baseline on {(mae < naive).sum()}/7 dims")

    # ---------------- 2. does our env state land on the data manifold? ----------------
    print()
    print("=" * 72)
    print("2. STATE MATCH: env-built state vs nearest dataset frame of the same task")
    print("=" * 72)
    from libero.libero import benchmark, get_libero_path

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task_id)
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
    ours = build_state(raw)
    print(f"env task: {task.language!r}")
    print(f"our state : {ours.round(4)}")

    d = meta.episodes.to_pandas()
    d["task"] = d["tasks"].apply(lambda t: str(t[0]))
    eps = d[d["task"] == task.language]["episode_index"].tolist()[:8]
    print(f"comparing against {len(eps)} dataset episodes of this task")
    ds = LeRobotDataset(args.dataset, episodes=eps)
    S = np.stack([ds[i]["observation.state"].numpy() for i in range(len(ds))])
    dist = np.linalg.norm(S - ours[None], axis=1)
    j = int(dist.argmin())
    print(f"nearest of {len(S)} frames: index {j}, L2 distance {dist[j]:.4f}")
    print(f"that frame: {S[j].round(4)}")
    print(f"per-dim gap: {(S[j] - ours).round(4)}")
    print(f"dataset state std: {np.asarray(meta.stats['observation.state']['std']).round(4)}")
    print("\n>>> a distance well under the dataset std means the state construction is right")

    # ---------------- 3. images, anchored to that matched frame ----------------
    print()
    print("=" * 72)
    print("3. IMAGE MATCH against the nearest frame")
    print("=" * 72)
    ref = ds[j]
    for cam, key in (("agentview_image", "observation.images.image"),
                     ("robot0_eye_in_hand_image", "observation.images.wrist_image")):
        r = ref[key].float().numpy()
        print(f"{cam}:")
        for name, fn in IMAGE_TRANSFORMS.items():
            c = to_chw01(fn(raw[cam])).numpy()
            print(f"   {name:>7}: MSE={float(((c - r) ** 2).mean()):.5f}")
    # ---------------- 4. are the recorded actions really OSC deltas? ----------------
    print()
    print("=" * 72)
    print("4. ACTION CONVENTION: recorded action vs realised change in eef position")
    print("=" * 72)
    # If action[:3] is a normalized OSC_POSE delta then the realised change in eef position
    # over one control step should be proportional to it, with slope near the controller's
    # output_max of 0.05. A near-zero correlation would mean env.step(action) is simply not
    # the right call -- absolute pose, joint space, or a different scaling.
    one = LeRobotDataset(args.dataset, episodes=[eps[0]])
    S1 = np.stack([one[i]["observation.state"].numpy() for i in range(len(one))])
    A1 = np.stack([one[i]["action"].numpy() for i in range(len(one))])
    if A1.ndim == 3:
        A1 = A1[:, 0]
    dpos, cmd = np.diff(S1[:, :3], axis=0), A1[:-1, :3]
    for k, nm in enumerate("xyz"):
        r = np.corrcoef(dpos[:, k], cmd[:, k])[0, 1]
        slope = np.polyfit(cmd[:, k], dpos[:, k], 1)[0]
        print(f"  d{nm}: corr={r:+.3f}  slope={slope:+.4f}  (OSC output_max=0.05)")
    print(f"  gripper action values: {np.unique(np.round(A1[:, 6], 2))[:8]}")

    env.close()


if __name__ == "__main__":
    main()
