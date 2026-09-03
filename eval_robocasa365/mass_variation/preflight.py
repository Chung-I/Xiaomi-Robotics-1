# Copyright (C) 2026 Xiaomi Corporation.
"""Task 1, Step 3: preflight validator for the mass/CoM study's candidate cells.

Sim-touching script (kept out of conditions.py, which must import without
robocasa). Validates:

  - primary cell: (PickPlaceCounterToCabinet, milk)
  - secondary candidates: (PickPlaceToasterToCounter, <category>) for each of
    bottled_water / boxed_food / bottled_drink / boxed_drink / canned_food

For each candidate: 3 seeds to check the object pool is non-empty (no
ValueError) *and* that the requested category is actually the one sampled
(some env classes hardcode their object cfg's obj_groups and silently ignore
the env-level kwarg -- this is exactly the kind of real, sim-data outcome
this script exists to catch, per the task ruling "pool ValueError is DATA,
not a bug"). If a candidate passes both, 10 seeds check mesh diversity
(>= 5 distinct meshes). The secondary choice is the first candidate, in
list order, that passes every check.

Per Plan amendment A, Phase 1 runs only the primary cell -- this script still
validates the secondary candidates for the deferred arm.

Run with: MUJOCO_GL=egl ~/Codes/robocasa/.venv/bin/python eval_robocasa365/mass_variation/preflight.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = REPO_ROOT / "output" / "mass_variation" / "preflight.json"

PRIMARY_ENV = "PickPlaceCounterToCabinet"
PRIMARY_CATEGORY = "milk"

SECONDARY_ENV = "PickPlaceToasterToCounter"
SECONDARY_CANDIDATES = (
    "bottled_water",
    "boxed_food",
    "bottled_drink",
    "boxed_drink",
    "canned_food",
)

POOL_CHECK_SEEDS = (0, 1, 2)
DIVERSITY_CHECK_SEEDS = tuple(range(10))
MIN_DISTINCT_MESHES = 5


def _mesh_category_from_path(mjcf_path: str) -> str | None:
    """Category segment of an asset path: .../objects/<registry>/<category>/<mesh_id>/model*.xml."""
    parts = Path(mjcf_path).parts
    try:
        idx = parts.index("objects")
    except ValueError:
        return None
    if idx + 2 >= len(parts):
        return None
    return parts[idx + 2]


def _mesh_id_from_path(mjcf_path: str) -> str:
    parts = Path(mjcf_path).parts
    try:
        idx = parts.index("objects")
    except ValueError:
        return mjcf_path
    if idx + 3 < len(parts):
        return parts[idx + 3]
    return mjcf_path


def probe_cell(
    gym: Any,
    get_task_horizon: Any,
    env_name: str,
    category: str,
    seeds: tuple[int, ...],
) -> dict:
    """Construct env_name with obj_groups=category and reset it for each seed.

    Records real outcomes as data: an empty-pool ValueError is not treated
    as a bug, and neither is a mismatch between the requested category and
    the category actually sampled (env classes that hardcode their object
    cfg silently ignore the obj_groups kwarg).
    """
    result: dict[str, Any] = {
        "env_name": env_name,
        "category": category,
        "n_seeds": len(seeds),
        "horizon": None,
        "seeds": [],
        "pool_nonempty": True,
        "category_respected": True,
        "distinct_mesh_ids": [],
    }

    try:
        result["horizon"] = get_task_horizon(env_name)
    except Exception as error:  # pragma: no cover - registry lookup
        result["horizon_error"] = repr(error)

    try:
        env = gym.make(
            f"robocasa/{env_name}",
            split="pretrain",
            obj_groups=category,
            seed=seeds[0],
        )
    except Exception as error:
        result["pool_nonempty"] = False
        result["category_respected"] = False
        result["construction_error"] = repr(error)
        return result

    try:
        for seed in seeds:
            seed_record: dict[str, Any] = {"seed": seed}
            try:
                env.reset(seed=seed)
                inner = env.unwrapped
                obj = inner.objects["obj"]
                bid = inner.sim.model.body_name2id(obj.root_body)
                mass_kg = float(inner.sim.model.body_mass[bid])
                mjcf_path = str(obj.mjcf_path)
                mesh_category = _mesh_category_from_path(mjcf_path)
                mesh_id = _mesh_id_from_path(mjcf_path)
                category_match = mesh_category == category

                seed_record.update(
                    {
                        "mesh_id": mesh_id,
                        "mesh_category": mesh_category,
                        "mjcf_path": mjcf_path,
                        "default_body_mass_kg": mass_kg,
                        "category_match": category_match,
                    }
                )
                if not category_match:
                    result["category_respected"] = False
                if mesh_id not in result["distinct_mesh_ids"]:
                    result["distinct_mesh_ids"].append(mesh_id)
            except ValueError as error:
                # Empty (fixture, category) pool -- real sim data, not a bug.
                result["pool_nonempty"] = False
                seed_record["error"] = repr(error)
            result["seeds"].append(seed_record)
    finally:
        env.close()

    return result


def _passes(probe: dict) -> bool:
    return bool(probe["pool_nonempty"] and probe["category_respected"])


def main() -> None:
    import gymnasium as gym
    import robocasa  # noqa: F401
    from robocasa.utils.dataset_registry_utils import get_task_horizon

    primary = probe_cell(
        gym, get_task_horizon, PRIMARY_ENV, PRIMARY_CATEGORY, POOL_CHECK_SEEDS
    )

    secondary_candidates: list[dict] = []
    secondary_choice: str | None = None
    secondary_choice_reason = "no candidate passed pool + category-match + diversity checks"

    for category in SECONDARY_CANDIDATES:
        pool_probe = probe_cell(
            gym, get_task_horizon, SECONDARY_ENV, category, POOL_CHECK_SEEDS
        )
        candidate: dict[str, Any] = {"category": category, "pool_check": pool_probe}

        if _passes(pool_probe):
            diversity_probe = probe_cell(
                gym,
                get_task_horizon,
                SECONDARY_ENV,
                category,
                DIVERSITY_CHECK_SEEDS,
            )
            candidate["diversity_check"] = diversity_probe
            n_distinct = len(diversity_probe["distinct_mesh_ids"])
            candidate["distinct_mesh_count"] = n_distinct
            candidate["passes"] = (
                _passes(diversity_probe) and n_distinct >= MIN_DISTINCT_MESHES
            )
        else:
            candidate["passes"] = False
            if not pool_probe["category_respected"]:
                candidate["fail_reason"] = (
                    "requested obj_groups was not respected -- env sampled "
                    "mesh category "
                    f"{pool_probe['seeds'][0].get('mesh_category')!r} instead of "
                    f"{category!r} (see notes: PickPlaceToasterToCounter hardcodes "
                    "its 'obj' cfg's obj_groups)"
                )
            else:
                candidate["fail_reason"] = "object pool empty (ValueError on reset)"

        secondary_candidates.append(candidate)

        if secondary_choice is None and candidate["passes"]:
            secondary_choice = category
            secondary_choice_reason = (
                f"first candidate (list order) passing pool + category-match + "
                f">= {MIN_DISTINCT_MESHES} distinct meshes over "
                f"{len(DIVERSITY_CHECK_SEEDS)} seeds"
            )

    notes = [
        (
            "VERIFIED FINDING: PickPlaceToasterToCounter._get_obj_cfgs "
            "(robocasa/environments/kitchen/atomic/kitchen_pick_place.py:1034) "
            "hardcodes the 'obj' cfg's obj_groups to ('sandwich_bread',) and "
            "never reads self.obj_groups. The env-level obj_groups kwarg passed "
            "at gym.make() is accepted (no TypeError, self.obj_groups is set) "
            "but silently has no effect on which object is sampled: every "
            "candidate category below still samples a sandwich_bread mesh. "
            "The design doc (docs/studies/2026-09-03-mass-com-xr1-design.md) "
            "excludes PickPlaceDrawerToCounter for this exact reason "
            "('hardcodes its object groups') but does not flag "
            "PickPlaceToasterToCounter -- this preflight run adds that "
            "finding. Confirmed empirically: obj_groups='canned_food' on "
            "PickPlaceToasterToCounter reset to "
            ".../lightwheel/sandwich_bread/SandwichBread005/model_upright.xml."
        ),
        (
            "Category-match check: extracts the category path segment from "
            "each sampled object's mjcf_path "
            "(.../objects/<registry>/<category>/<mesh_id>/model*.xml) and "
            "compares it to the requested obj_groups value."
        ),
    ]

    report = {
        "primary": {"env_name": PRIMARY_ENV, "category": PRIMARY_CATEGORY, "probe": primary},
        "secondary_candidates": secondary_candidates,
        "secondary_choice": secondary_choice,
        "secondary_choice_reason": secondary_choice_reason,
        "notes": notes,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(f"Wrote {OUTPUT_PATH}")
    print(f"Primary cell pool_nonempty={primary['pool_nonempty']}")
    if secondary_choice is not None:
        print(f"Secondary choice: {SECONDARY_ENV} / {secondary_choice}")
    else:
        print(f"Secondary choice: NONE ({secondary_choice_reason})")
        print(
            "See 'notes' in the report: PickPlaceToasterToCounter ignores "
            "obj_groups for all 5 candidates."
        )


if __name__ == "__main__":
    main()
