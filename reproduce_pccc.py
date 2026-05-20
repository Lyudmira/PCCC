#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import shutil
import sqlite3
import statistics
import struct
import subprocess
import sys
import time
import urllib.request
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"
SCENES = ["Ballroom", "Barn", "Church", "Family", "Francis", "Horse", "Ignatius", "Museum"]
DATASET_REPO = "kairunwen/InstantSplat"
PAIR_ID_OFFSET = 2147483647


def info(message: str) -> None:
    print(message, flush=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def run_command(command: list[str], *, cwd: Path, log_path: Path, allow_failure: bool = False) -> bool:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        completed = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log.write(completed.stdout)
    if completed.returncode != 0 and not allow_failure:
        raise RuntimeError(f"Command failed with code {completed.returncode}; see {log_path}")
    return completed.returncode == 0


def download_url(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    info(f"download {url}")
    with urllib.request.urlopen(url) as response, dest.open("wb") as out:
        shutil.copyfileobj(response, out)
    return dest


def ensure_tanks_root(*, source_tanks_root: Path | None, work_dir: Path) -> Path:
    target = work_dir / "data/source/InstantSplat/Tanks"
    if target.exists() and usable_scene_dirs(target):
        info(f"data already present: {target}")
        return target

    if source_tanks_root is not None:
        if not source_tanks_root.exists():
            raise FileNotFoundError(source_tanks_root)
        if not usable_scene_dirs(source_tanks_root):
            raise RuntimeError(f"No usable scenes under {source_tanks_root}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            shutil.rmtree(target)
        info(f"copy local Tanks: {source_tanks_root} -> {target}")
        shutil.copytree(source_tanks_root, target)
        return target

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Install dependencies first: python -m pip install -r requirements.txt") from exc

    archive_dir = work_dir / "data/downloads"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / "InstantSplat.zip"
    if not archive_path.exists():
        info(f"download official dataset: {DATASET_REPO}/InstantSplat.zip")
        downloaded = hf_hub_download(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            filename="InstantSplat.zip",
            local_dir=archive_dir,
            local_dir_use_symlinks=False,
        )
        downloaded_path = Path(downloaded)
        if downloaded_path != archive_path:
            shutil.copy2(downloaded_path, archive_path)

    extract_root = work_dir / "data/source"
    extract_root.mkdir(parents=True, exist_ok=True)
    info(f"extract {archive_path}")
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(extract_root)
    if not usable_scene_dirs(target):
        raise RuntimeError(f"Dataset extracted but no usable scenes found under {target}")
    return target


def latest_colmap_windows_asset(prefer_nocuda: bool = True) -> tuple[str, str]:
    url = "https://api.github.com/repos/colmap/colmap/releases/latest"
    with urllib.request.urlopen(url) as response:
        release = json.loads(response.read().decode("utf-8"))
    assets = release.get("assets", [])
    candidates = []
    for asset in assets:
        name = str(asset.get("name", "")).lower()
        if name.endswith(".zip") and "windows" in name:
            candidates.append(asset)
    if not candidates:
        raise RuntimeError("No Windows COLMAP zip asset found in the latest COLMAP GitHub release.")
    if prefer_nocuda:
        for asset in candidates:
            if "nocuda" in str(asset.get("name", "")).lower():
                return str(asset["name"]), str(asset["browser_download_url"])
    return str(candidates[0]["name"]), str(candidates[0]["browser_download_url"])


def find_colmap_executable(root: Path) -> Path | None:
    for name in ["COLMAP.bat", "colmap.bat", "colmap.exe"]:
        found = list(root.rglob(name))
        if found:
            return found[0]
    return None


def ensure_colmap(*, raw: str | None, work_dir: Path) -> Path:
    if raw:
        path = Path(raw)
        if path.exists():
            return path
    env = os.environ.get("COLMAP_BIN")
    if env and Path(env).exists():
        return Path(env)
    found = shutil.which("colmap")
    if found:
        return Path(found)

    local_root = work_dir / "external/colmap"
    existing = find_colmap_executable(local_root)
    if existing:
        return existing

    if platform.system().lower() != "windows":
        raise RuntimeError("Set COLMAP_BIN or install colmap on PATH. Auto-download is only implemented for Windows.")

    name, url = latest_colmap_windows_asset(prefer_nocuda=True)
    archive = work_dir / "external/downloads" / name
    download_url(url, archive)
    reset_dir(local_root)
    info(f"extract COLMAP: {archive}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(local_root)
    executable = find_colmap_executable(local_root)
    if executable is None:
        raise RuntimeError(f"COLMAP downloaded but executable not found under {local_root}")
    return executable


@dataclass(frozen=True)
class Camera:
    model: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def focal(self) -> float:
        return 0.5 * (self.fx + self.fy)

    def to_json(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "focal": self.focal,
        }


def read_camera(cameras_path: Path) -> Camera:
    for line in cameras_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        model = parts[1].upper()
        width = int(parts[2])
        height = int(parts[3])
        params = [float(v) for v in parts[4:]]
        if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL"}:
            fx = fy = params[0]
            cx, cy = params[1], params[2]
        elif model == "PINHOLE":
            fx, fy, cx, cy = params[:4]
        else:
            raise ValueError(f"Unsupported camera model {model} in {cameras_path}")
        return Camera(model=model, width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)
    raise ValueError(f"No camera row found in {cameras_path}")


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = np.asarray(qvec, dtype=np.float64).reshape(4)
    return np.array(
        [
            [1.0 - 2.0 * qy * qy - 2.0 * qz * qz, 2.0 * qx * qy - 2.0 * qw * qz, 2.0 * qx * qz + 2.0 * qw * qy],
            [2.0 * qx * qy + 2.0 * qw * qz, 1.0 - 2.0 * qx * qx - 2.0 * qz * qz, 2.0 * qy * qz - 2.0 * qw * qx],
            [2.0 * qx * qz - 2.0 * qw * qy, 2.0 * qy * qz + 2.0 * qw * qx, 1.0 - 2.0 * qx * qx - 2.0 * qy * qy],
        ],
        dtype=np.float64,
    )


def read_pose_records(images_path: Path) -> dict[str, tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    lines = [line.strip() for line in images_path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
    records: dict[str, tuple[int, np.ndarray, np.ndarray, np.ndarray]] = {}
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        image_id = int(parts[0])
        qvec = np.array([float(v) for v in parts[1:5]], dtype=np.float64)
        tvec = np.array([float(v) for v in parts[5:8]], dtype=np.float64)
        name = parts[9]
        records[name] = (image_id, qvec, qvec_to_rotmat(qvec), tvec)
        i += 2
    return records


def image_names(scene_dir: Path) -> list[str]:
    return [p.name for p in sorted((scene_dir / "images").iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]


def usable_scene_dirs(tanks_root: Path, requested: list[str] | None = None) -> list[Path]:
    if not tanks_root.exists():
        return []
    wanted = set(requested) if requested else None
    rows = []
    for scene in sorted(tanks_root.iterdir()):
        if wanted is not None and scene.name not in wanted:
            continue
        if (scene / "images").exists() and (scene / "sparse/0/cameras.txt").exists() and (scene / "sparse/0/images.txt").exists():
            rows.append(scene)
    order = {name: i for i, name in enumerate(SCENES)}
    return sorted(rows, key=lambda p: order.get(p.name, 999))


def prepare_crops_and_manifest(*, tanks_root: Path, work_dir: Path, scenes: list[str], counts: list[int], crop_size: int) -> Path:
    max_count = max(counts)
    crop_root = work_dir / "data/crops"
    rows = []
    for scene_dir in usable_scene_dirs(tanks_root, scenes):
        camera = read_camera(scene_dir / "sparse/0/cameras.txt")
        poses = read_pose_records(scene_dir / "sparse/0/images.txt")
        names = image_names(scene_dir)[:max_count]
        missing = [name for name in names if name not in poses]
        if missing:
            raise RuntimeError(f"{scene_dir.name}: missing poses for {missing[:3]}")
        out_dir = crop_root / scene_dir.name / "images"
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            src = scene_dir / "images" / name
            dst = out_dir / name
            if dst.exists():
                continue
            with Image.open(src) as im:
                if im.width < crop_size or im.height < crop_size:
                    raise RuntimeError(f"Image too small for {crop_size} crop: {src}")
                im.convert("RGB").crop((0, 0, crop_size, crop_size)).save(dst, quality=95)
        rows.append(
            {
                "scene": scene_dir.name,
                "source_scene_dir": str(scene_dir),
                "crop_image_dir": str(out_dir),
                "selected_images": names,
                "counts": counts,
                "crop": {"origin": [0, 0], "width": crop_size, "height": crop_size, "resize": False},
                "camera": camera.to_json(),
                "true_crop_principal_point": {"cx": camera.cx, "cy": camera.cy},
            }
        )
    manifest = {"dataset": DATASET_REPO, "tanks_root": str(tanks_root), "crop_root": str(crop_root), "scenes": rows}
    out = work_dir / "data/manifest.json"
    write_json(out, manifest)
    return out


def make_k(focal: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[float(focal), 0.0, float(cx)], [0.0, float(focal), float(cy)], [0.0, 0.0, 1.0]], dtype=np.float64)


def skew(t: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(t, dtype=np.float64).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def normalize_fro(matrix: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(matrix, ord="fro"))
    if norm <= 1e-15 or not np.isfinite(norm):
        raise ValueError("Cannot normalize matrix.")
    return matrix / norm


def fundamental_from_rt(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(t))
    if norm <= 1e-12 or not np.isfinite(norm):
        t = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        t = t / norm
    k_inv = np.linalg.inv(K)
    return normalize_fro(k_inv.T @ skew(t) @ np.asarray(R, dtype=np.float64).reshape(3, 3) @ k_inv)


def homogeneous(points: np.ndarray) -> np.ndarray:
    return np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)


def sampson_signed_residuals(points0: np.ndarray, points1: np.ndarray, F: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x0 = homogeneous(np.asarray(points0, dtype=np.float64))
    x1 = homogeneous(np.asarray(points1, dtype=np.float64))
    Fx0 = x0 @ F.T
    Ftx1 = x1 @ F
    numerator = np.sum(x1 * Fx0, axis=1)
    denominator = Fx0[:, 0] ** 2 + Fx0[:, 1] ** 2 + Ftx1[:, 0] ** 2 + Ftx1[:, 1] ** 2
    return numerator / np.sqrt(np.maximum(denominator, eps))


def seq_pairs(names: list[str]) -> list[tuple[str, str]]:
    return [(names[i], names[i + 1]) for i in range(len(names) - 1)]


def all_pairs(names: list[str]) -> list[tuple[str, str]]:
    return [(a, b) for i, a in enumerate(names) for b in names[i + 1 :]]


def detect_sift_matches(crop_dir: Path, pairs: list[tuple[str, str]], *, max_features: int, ratio_test: float) -> tuple[str, dict[tuple[str, str], tuple[np.ndarray, np.ndarray]], dict[str, int]]:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("OpenCV is required. Install with: python -m pip install -r requirements.txt") from exc

    names = sorted({name for pair in pairs for name in pair})
    images = {name: cv2.imread(str(crop_dir / name), cv2.IMREAD_GRAYSCALE) for name in names}
    missing = [name for name, image in images.items() if image is None]
    if missing:
        raise FileNotFoundError(f"Failed to load crop images: {missing[:3]}")

    if hasattr(cv2, "SIFT_create"):
        detector_name = "SIFT"
        detector = cv2.SIFT_create(nfeatures=max_features)
        norm = cv2.NORM_L2
    else:
        detector_name = "ORB"
        detector = cv2.ORB_create(nfeatures=max_features)
        norm = cv2.NORM_HAMMING

    features = {name: detector.detectAndCompute(image, None) for name, image in images.items()}
    matcher = cv2.BFMatcher(norm)
    matches: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    counts: dict[str, int] = {}
    for first, second in pairs:
        kp1, desc1 = features[first]
        kp2, desc2 = features[second]
        if desc1 is None or desc2 is None:
            pts1 = np.zeros((0, 2), dtype=np.float64)
            pts2 = np.zeros((0, 2), dtype=np.float64)
        else:
            good = []
            for pair in matcher.knnMatch(desc1, desc2, k=2):
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < ratio_test * n.distance:
                    good.append(m)
            pts1 = np.array([kp1[m.queryIdx].pt for m in good], dtype=np.float64)
            pts2 = np.array([kp2[m.trainIdx].pt for m in good], dtype=np.float64)
        matches[(first, second)] = (pts1, pts2)
        counts[f"{first}->{second}"] = int(pts1.shape[0])
    return detector_name, matches, counts


def relative_pose(records: dict[str, tuple[int, np.ndarray, np.ndarray, np.ndarray]], first: str, second: str) -> tuple[np.ndarray, np.ndarray]:
    _id1, _q1, R1, t1 = records[first]
    _id2, _q2, R2, t2 = records[second]
    R = R2 @ R1.T
    t = t2 - R @ t1
    return R, t


def build_joint_objective(
    *,
    records: dict[str, tuple[int, np.ndarray, np.ndarray, np.ndarray]],
    matches: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    trim_quantile: float,
) -> tuple[Any, int]:
    terms = []
    for (first, second), (pts1, pts2) in matches.items():
        if pts1.shape[0] < 8:
            continue
        R, t = relative_pose(records, first, second)
        terms.append((pts1, pts2, R, t))
    if not terms:
        raise RuntimeError("No image pair has at least 8 matches.")

    def objective(x: np.ndarray | tuple[float, float, float]) -> float:
        cx, cy, focal = float(x[0]), float(x[1]), float(x[2])
        if focal <= 1e-6 or not np.isfinite(focal):
            return 1e9
        K = make_k(focal, cx, cy)
        chunks = []
        for pts1, pts2, R, t in terms:
            F = fundamental_from_rt(K, R, t)
            chunks.append(np.abs(sampson_signed_residuals(pts1, pts2, F)))
        residuals = np.concatenate(chunks)
        cutoff = float(np.quantile(residuals, trim_quantile))
        trimmed = residuals[residuals <= cutoff]
        return float(np.mean(trimmed))

    return objective, len(terms)


def blind_joint_search(
    objective: Any,
    *,
    bounds_cx: tuple[float, float],
    bounds_cy: tuple[float, float],
    focal_bounds: tuple[float, float],
    grid_step_px: float,
    focal_step_px: float,
    num_refine_seeds: int,
    refine_maxiter: int,
) -> dict[str, Any]:
    xs = np.arange(bounds_cx[0], bounds_cx[1] + 0.5 * grid_step_px, grid_step_px, dtype=np.float64)
    ys = np.arange(bounds_cy[0], bounds_cy[1] + 0.5 * grid_step_px, grid_step_px, dtype=np.float64)
    fs = np.arange(focal_bounds[0], focal_bounds[1] + 0.5 * focal_step_px, focal_step_px, dtype=np.float64)
    coarse = []
    for f in fs:
        for cy in ys:
            for cx in xs:
                coarse.append({"cx": float(cx), "cy": float(cy), "focal": float(f), "value": float(objective((cx, cy, f)))})
    coarse.sort(key=lambda row: row["value"])
    refined = []
    for seed in coarse[:num_refine_seeds]:
        result = minimize(
            lambda values: objective(values),
            x0=np.array([seed["cx"], seed["cy"], seed["focal"]], dtype=np.float64),
            method="Powell",
            bounds=[bounds_cx, bounds_cy, focal_bounds],
            options={"maxiter": refine_maxiter, "xtol": 1e-3, "ftol": 1e-6},
        )
        refined.append(
            {
                "start": seed,
                "cx": float(result.x[0]),
                "cy": float(result.x[1]),
                "focal": float(result.x[2]),
                "value": float(result.fun),
                "success": bool(result.success),
                "message": str(result.message),
                "nit": int(getattr(result, "nit", -1)),
            }
        )
    refined.sort(key=lambda row: row["value"])
    return {"best": refined[0], "coarse_top5": coarse[:5], "refined_top5": refined[:5], "num_grid": len(coarse)}


def run_joint_focal(
    *,
    manifest_path: Path,
    output_dir: Path,
    scenes: list[str],
    counts: list[int],
    max_features: int,
    ratio_test: float,
    trim_quantile: float,
) -> list[dict[str, Any]]:
    manifest = read_json(manifest_path)
    rows: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for scene in manifest["scenes"]:
        scene_name = scene["scene"]
        if scene_name not in scenes:
            continue
        info(f"=== joint focal {scene_name} ===")
        crop_dir = Path(scene["crop_image_dir"])
        source_dir = Path(scene["source_scene_dir"])
        records = read_pose_records(source_dir / "sparse/0/images.txt")
        true_cx = float(scene["true_crop_principal_point"]["cx"])
        true_cy = float(scene["true_crop_principal_point"]["cy"])
        image_names_all = [str(n) for n in scene["selected_images"]]
        for count in counts:
            names = image_names_all[:count]
            pairs = seq_pairs(names)
            detector, matches, match_counts = detect_sift_matches(crop_dir, pairs, max_features=max_features, ratio_test=ratio_test)
            objective, num_pairs_used = build_joint_objective(records=records, matches=matches, trim_quantile=trim_quantile)
            start = time.perf_counter()
            search = blind_joint_search(
                objective,
                bounds_cx=(-240.0, 960.0),
                bounds_cy=(-240.0, 720.0),
                focal_bounds=(350.0, 850.0),
                grid_step_px=40.0,
                focal_step_px=50.0,
                num_refine_seeds=5,
                refine_maxiter=120,
            )
            elapsed = time.perf_counter() - start
            best = search["best"]
            pp = math.hypot(float(best["cx"]) - true_cx, float(best["cy"]) - true_cy)
            row = {
                "scene": scene_name,
                "count": int(count),
                "pair_policy": "seq",
                "detector": detector,
                "num_pairs_requested": len(pairs),
                "num_pairs_used": int(num_pairs_used),
                "pair_match_counts": match_counts,
                "cx": float(best["cx"]),
                "cy": float(best["cy"]),
                "focal_px": float(best["focal"]),
                "objective": float(best["value"]),
                "pp_error_px": float(pp),
                "elapsed_seconds": float(elapsed),
                "coarse_top5": search["coarse_top5"],
                "refined_top5": search["refined_top5"],
            }
            rows.append(row)
            write_json(output_dir / "results.json", rows)
            info(f"{scene_name:9s} N={count:2d} pp={pp:.3f} f={best['focal']:.3f} obj={best['value']:.9f}")
    write_json(output_dir / "results.json", rows)
    return rows


def image_pair_to_pair_id(image_id1: int, image_id2: int) -> int:
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return PAIR_ID_OFFSET * int(image_id1) + int(image_id2)


def create_colmap_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE cameras (
            camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            model INTEGER NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            params BLOB,
            prior_focal_length INTEGER NOT NULL
        );
        CREATE TABLE images (
            image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            name TEXT NOT NULL UNIQUE,
            camera_id INTEGER NOT NULL,
            prior_qw REAL, prior_qx REAL, prior_qy REAL, prior_qz REAL,
            prior_tx REAL, prior_ty REAL, prior_tz REAL,
            CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < 2147483647),
            FOREIGN KEY(camera_id) REFERENCES cameras(camera_id)
        );
        CREATE UNIQUE INDEX index_name ON images(name);
        CREATE TABLE keypoints (image_id INTEGER PRIMARY KEY NOT NULL, rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB);
        CREATE TABLE descriptors (image_id INTEGER PRIMARY KEY NOT NULL, rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB);
        CREATE TABLE matches (pair_id INTEGER PRIMARY KEY NOT NULL, rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB);
        CREATE TABLE two_view_geometries (
            pair_id INTEGER PRIMARY KEY NOT NULL,
            rows INTEGER NOT NULL,
            cols INTEGER NOT NULL,
            data BLOB,
            config INTEGER NOT NULL,
            F BLOB, E BLOB, H BLOB, qvec BLOB, tvec BLOB
        );
        """
    )


def keypoints_to_colmap(keypoints: list[Any]) -> np.ndarray:
    out = np.zeros((len(keypoints), 6), dtype=np.float32)
    for i, kp in enumerate(keypoints):
        out[i, 0] = float(kp.pt[0])
        out[i, 1] = float(kp.pt[1])
        out[i, 2] = 1.0
        out[i, 5] = 1.0
    return out


def descriptors_to_colmap(desc: np.ndarray | None, rows: int) -> np.ndarray:
    if desc is None or rows == 0:
        return np.empty((0, 128), dtype=np.uint8)
    if desc.shape[1] != 128:
        return np.zeros((rows, 128), dtype=np.uint8)
    if desc.dtype == np.uint8:
        return desc.copy()
    return np.clip(np.rint(desc), 0, 255).astype(np.uint8)


@dataclass(frozen=True)
class ColmapDbStats:
    database_path: str
    detector: str
    num_images: int
    num_pairs_requested: int
    num_pairs_written: int
    num_pairs_skipped_too_few_matches: int
    num_keypoints: int
    num_matches: int
    min_matches_per_pair: int
    max_matches_per_pair: int
    mean_matches_per_pair: float


def build_colmap_database(
    *,
    crop_dir: Path,
    database_path: Path,
    names: list[str],
    policy: str,
    width: int,
    height: int,
    initial_focal: float,
    initial_cx: float,
    initial_cy: float,
    max_features: int,
    ratio_test: float,
    min_matches: int,
) -> ColmapDbStats:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("OpenCV is required. Install with: python -m pip install -r requirements.txt") from exc

    images = {name: cv2.imread(str(crop_dir / name), cv2.IMREAD_GRAYSCALE) for name in names}
    missing = [name for name, image in images.items() if image is None]
    if missing:
        raise FileNotFoundError(f"Failed to load crop images: {missing[:3]}")

    if hasattr(cv2, "SIFT_create"):
        detector_name = "SIFT"
        detector = cv2.SIFT_create(nfeatures=max_features)
        norm = cv2.NORM_L2
    else:
        detector_name = "ORB"
        detector = cv2.ORB_create(nfeatures=max_features)
        norm = cv2.NORM_HAMMING

    features = {}
    for name, image in images.items():
        kps, desc = detector.detectAndCompute(image, None)
        features[name] = (kps, desc, keypoints_to_colmap(kps), descriptors_to_colmap(desc, len(kps)))

    pairs = seq_pairs(names) if policy == "seq" else all_pairs(names)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    if database_path.exists():
        database_path.unlink()
    conn = sqlite3.connect(database_path)
    counts: list[int] = []
    skipped = 0
    try:
        create_colmap_schema(conn)
        params = struct.pack("<dddd", float(initial_focal), float(initial_focal), float(initial_cx), float(initial_cy))
        conn.execute(
            "INSERT INTO cameras(camera_id, model, width, height, params, prior_focal_length) VALUES(1, 1, ?, ?, ?, 0)",
            (int(width), int(height), params),
        )
        id_by_name = {name: i + 1 for i, name in enumerate(names)}
        for name, image_id in id_by_name.items():
            _kps, _desc, kp_arr, desc_arr = features[name]
            conn.execute("INSERT INTO images(image_id, name, camera_id) VALUES(?, ?, 1)", (image_id, name))
            conn.execute("INSERT INTO keypoints(image_id, rows, cols, data) VALUES(?, ?, 6, ?)", (image_id, kp_arr.shape[0], np.asarray(kp_arr, dtype="<f4").tobytes()))
            conn.execute("INSERT INTO descriptors(image_id, rows, cols, data) VALUES(?, ?, 128, ?)", (image_id, desc_arr.shape[0], np.asarray(desc_arr, dtype=np.uint8).tobytes()))

        matcher = cv2.BFMatcher(norm)
        for first, second in pairs:
            kp1, desc1, _kp_arr1, _desc_arr1 = features[first]
            kp2, desc2, _kp_arr2, _desc_arr2 = features[second]
            if desc1 is None or desc2 is None:
                match_idx = np.empty((0, 2), dtype=np.uint32)
            else:
                good = []
                for pair in matcher.knnMatch(desc1, desc2, k=2):
                    if len(pair) < 2:
                        continue
                    m, n = pair
                    if m.distance < ratio_test * n.distance:
                        good.append(m)
                match_idx = np.asarray([[m.queryIdx, m.trainIdx] for m in good], dtype=np.uint32).reshape(-1, 2)
            if match_idx.shape[0] < min_matches:
                skipped += 1
                continue
            id1 = id_by_name[first]
            id2 = id_by_name[second]
            if id1 > id2:
                match_idx = match_idx[:, ::-1].copy()
            pair_id = image_pair_to_pair_id(id1, id2)
            blob = np.asarray(match_idx, dtype="<u4").reshape(-1).tobytes()
            conn.execute("INSERT INTO matches(pair_id, rows, cols, data) VALUES(?, ?, 2, ?)", (pair_id, match_idx.shape[0], blob))
            conn.execute(
                "INSERT INTO two_view_geometries(pair_id, rows, cols, data, config, F, E, H, qvec, tvec) VALUES(?, ?, 2, ?, 3, NULL, NULL, NULL, NULL, NULL)",
                (pair_id, match_idx.shape[0], blob),
            )
            counts.append(int(match_idx.shape[0]))
        conn.commit()
    finally:
        conn.close()

    num_keypoints = sum(features[name][2].shape[0] for name in names)
    return ColmapDbStats(
        database_path=str(database_path),
        detector=detector_name,
        num_images=len(names),
        num_pairs_requested=len(pairs),
        num_pairs_written=len(counts),
        num_pairs_skipped_too_few_matches=skipped,
        num_keypoints=int(num_keypoints),
        num_matches=int(sum(counts)),
        min_matches_per_pair=int(min(counts)) if counts else 0,
        max_matches_per_pair=int(max(counts)) if counts else 0,
        mean_matches_per_pair=float(np.mean(counts)) if counts else 0.0,
    )


def write_known_rt_model(
    *,
    model_dir: Path,
    names: list[str],
    records: dict[str, tuple[int, np.ndarray, np.ndarray, np.ndarray]],
    width: int,
    height: int,
    focal: float,
    cx: float,
    cy: float,
) -> None:
    reset_dir(model_dir)
    (model_dir / "cameras.txt").write_text(
        f"# Camera list\n1 PINHOLE {width} {height} {focal:.17g} {focal:.17g} {cx:.17g} {cy:.17g}\n",
        encoding="utf-8",
    )
    lines = ["# Image list", f"# Number of images: {len(names)}, mean observations per image: 0"]
    for image_id, name in enumerate(names, start=1):
        if name not in records:
            raise RuntimeError(f"Missing pose for {name}")
        _src_id, qvec, _R, tvec = records[name]
        vals = [str(image_id), *[f"{float(v):.17g}" for v in qvec], *[f"{float(v):.17g}" for v in tvec], "1", name]
        lines.append(" ".join(vals))
        lines.append("")
    (model_dir / "images.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (model_dir / "points3D.txt").write_text("# 3D point list\n# Number of points: 0, mean track length: 0\n", encoding="utf-8")


def read_count_from_header(path: Path, pattern: str) -> int:
    regex = re.compile(pattern)
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = regex.search(line)
        if match:
            return int(match.group(1))
    return 0


def convert_model(colmap_bin: Path, model_path: Path, output_path: Path) -> tuple[int, int, Path]:
    reset_dir(output_path)
    run_command(
        [str(colmap_bin), "model_converter", "--input_path", str(model_path), "--output_path", str(output_path), "--output_type", "TXT", "--log_level", "2"],
        cwd=ROOT,
        log_path=output_path / "model_converter.log",
    )
    registered = read_count_from_header(output_path / "images.txt", r"Number of images: (\d+)")
    points = read_count_from_header(output_path / "points3D.txt", r"Number of points: (\d+)")
    return registered, points, output_path


def read_colmap_camera(cameras_path: Path) -> dict[str, Any]:
    for line in cameras_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        model = parts[1].upper()
        width = int(parts[2])
        height = int(parts[3])
        params = [float(v) for v in parts[4:]]
        if model == "PINHOLE":
            fx, fy, cx, cy = params[:4]
        elif model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL"}:
            fx = fy = params[0]
            cx, cy = params[1], params[2]
        else:
            fx = fy = cx = cy = math.nan
        return {"width": width, "height": height, "model": model, "params": params, "fx": fx, "fy": fy, "focal": 0.5 * (fx + fy), "cx": cx, "cy": cy}
    raise RuntimeError(f"No camera row in {cameras_path}")


def run_known_rt_colmap(
    *,
    colmap_bin: Path,
    database_path: Path,
    crop_dir: Path,
    input_model: Path,
    work_dir: Path,
    true_cx: float,
    true_cy: float,
) -> dict[str, Any]:
    triangulated = work_dir / "triangulated"
    bundle_adjusted = work_dir / "bundle_adjusted"
    reset_dir(triangulated)
    reset_dir(bundle_adjusted)
    common = ["--default_random_seed", "0", "--log_level", "2"]
    ok_tri = run_command(
        [
            str(colmap_bin),
            "point_triangulator",
            "--database_path", str(database_path.resolve()),
            "--image_path", str(crop_dir.resolve()),
            "--input_path", str(input_model.resolve()),
            "--output_path", str(triangulated.resolve()),
            "--clear_points", "1",
            "--refine_intrinsics", "0",
            "--Mapper.fix_existing_frames", "1",
            "--Mapper.tri_ignore_two_view_tracks", "0",
            "--Mapper.tri_min_angle", "0.1",
            "--Mapper.filter_max_reproj_error", "8",
            *common,
        ],
        cwd=ROOT,
        log_path=work_dir / "point_triangulator.log",
        allow_failure=True,
    )
    if not ok_tri:
        return {"status": "triangulator_failed", "registered_images": 0, "points3D": 0, "intrinsics": {"principal_error_px": math.nan}}

    ok_ba = run_command(
        [
            str(colmap_bin),
            "bundle_adjuster",
            "--input_path", str(triangulated.resolve()),
            "--output_path", str(bundle_adjusted.resolve()),
            "--BundleAdjustment.refine_focal_length", "1",
            "--BundleAdjustment.refine_principal_point", "1",
            "--BundleAdjustment.refine_extra_params", "0",
            "--BundleAdjustment.refine_rig_from_world", "0",
            "--BundleAdjustment.refine_sensor_from_rig", "0",
            "--BundleAdjustment.refine_points3D", "1",
            "--BundleAdjustmentCeres.max_num_iterations", "100",
            *common,
        ],
        cwd=ROOT,
        log_path=work_dir / "bundle_adjuster.log",
        allow_failure=True,
    )
    final_model = bundle_adjusted if ok_ba else triangulated
    registered, points, txt = convert_model(colmap_bin, final_model, work_dir / ("txt_bundle_adjusted" if ok_ba else "txt_triangulated"))
    intr = read_colmap_camera(txt / "cameras.txt")
    intr["principal_error_px"] = math.hypot(float(intr["cx"]) - true_cx, float(intr["cy"]) - true_cy)
    return {
        "status": "ok" if ok_ba else "ba_failed_used_triangulated",
        "triangulator_ok": bool(ok_tri),
        "bundle_adjuster_ok": bool(ok_ba),
        "registered_images": int(registered),
        "points3D": int(points),
        "intrinsics": intr,
        "model_path": str(final_model),
    }


def joint_error_lookup(joint_rows: list[dict[str, Any]]) -> dict[tuple[str, int], float]:
    return {(row["scene"], int(row["count"])): float(row["pp_error_px"]) for row in joint_rows}


def run_colmap_baseline(
    *,
    manifest_path: Path,
    output_dir: Path,
    colmap_bin: Path,
    policy: str,
    scenes: list[str],
    counts: list[int],
    joint_rows: list[dict[str, Any]],
    max_features: int,
    ratio_test: float,
) -> list[dict[str, Any]]:
    manifest = read_json(manifest_path)
    ours = joint_error_lookup(joint_rows)
    rows: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for scene in manifest["scenes"]:
        scene_name = scene["scene"]
        if scene_name not in scenes:
            continue
        info(f"=== COLMAP known-RT {policy} {scene_name} ===")
        crop_dir = Path(scene["crop_image_dir"])
        source_dir = Path(scene["source_scene_dir"])
        records = read_pose_records(source_dir / "sparse/0/images.txt")
        true_cx = float(scene["true_crop_principal_point"]["cx"])
        true_cy = float(scene["true_crop_principal_point"]["cy"])
        width = int(scene["crop"]["width"])
        height = int(scene["crop"]["height"])
        initial_focal = 1.2 * max(width, height)
        initial_cx = 0.5 * width
        initial_cy = 0.5 * height
        image_names_all = [str(n) for n in scene["selected_images"]]
        for count in counts:
            names = image_names_all[:count]
            work_dir = output_dir / "work" / scene_name / f"N{count}_{policy}"
            reset_dir(work_dir)
            db_path = work_dir / "database.db"
            db = build_colmap_database(
                crop_dir=crop_dir,
                database_path=db_path,
                names=names,
                policy=policy,
                width=width,
                height=height,
                initial_focal=initial_focal,
                initial_cx=initial_cx,
                initial_cy=initial_cy,
                max_features=max_features,
                ratio_test=ratio_test,
                min_matches=8,
            )
            input_model = work_dir / "input_known_rt_model"
            write_known_rt_model(model_dir=input_model, names=names, records=records, width=width, height=height, focal=initial_focal, cx=initial_cx, cy=initial_cy)
            colmap_result = run_known_rt_colmap(
                colmap_bin=colmap_bin,
                database_path=db_path,
                crop_dir=crop_dir,
                input_model=input_model,
                work_dir=work_dir,
                true_cx=true_cx,
                true_cy=true_cy,
            )
            row = {
                "scene": scene_name,
                "num_images": int(count),
                "pair_policy": policy,
                "image_names": names,
                "true_cx": true_cx,
                "true_cy": true_cy,
                "initial_camera": {
                    "model": "PINHOLE",
                    "width": width,
                    "height": height,
                    "focal": initial_focal,
                    "cx": initial_cx,
                    "cy": initial_cy,
                    "principal_error_px": math.hypot(initial_cx - true_cx, initial_cy - true_cy),
                },
                "database": asdict(db),
                "colmap": colmap_result,
                "ours_joint_pp_error_px": ours.get((scene_name, count), math.nan),
            }
            rows.append(row)
            write_json(output_dir / "results.json", rows)
            err = colmap_result["intrinsics"].get("principal_error_px", math.nan)
            info(f"{scene_name:9s} N={count:2d} pairs={db.num_pairs_written}/{db.num_pairs_requested} COLMAP={colmap_result['status']} pp={err}")
    write_json(output_dir / "results.json", rows)
    return rows


def stats(values: list[float]) -> dict[str, float]:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return {"mean": math.nan, "median": math.nan, "min": math.nan, "max": math.nan}
    return {"mean": float(statistics.fmean(finite)), "median": float(statistics.median(finite)), "min": float(min(finite)), "max": float(max(finite))}


def fmt(value: Any, digits: int = 3) -> str:
    try:
        x = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(x):
        return "nan"
    return f"{x:.{digits}f}"


def write_tables(*, output_dir: Path, joint_rows: list[dict[str, Any]], colmap_seq: list[dict[str, Any]] | None, colmap_all: list[dict[str, Any]] | None) -> None:
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    lines = ["# PCCC joint focal N ablation", "", "| N | runs | mean pp | median pp | min pp | max pp | mean focal |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for count in sorted({int(r["count"]) for r in joint_rows}):
        rows = [r for r in joint_rows if int(r["count"]) == count]
        pp = stats([r["pp_error_px"] for r in rows])
        focal = stats([r["focal_px"] for r in rows])
        lines.append(f"| {count} | {len(rows)} | {fmt(pp['mean'])} | {fmt(pp['median'])} | {fmt(pp['min'])} | {fmt(pp['max'])} | {fmt(focal['mean'])} |")
    (table_dir / "joint_focal_n_ablation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    n24 = [r for r in joint_rows if int(r["count"]) == 24]
    lines = ["# PCCC N=24 per-scene joint focal", "", "| scene | cx | cy | focal | pp err | objective |", "|---|---:|---:|---:|---:|---:|"]
    for r in n24:
        lines.append(f"| {r['scene']} | {fmt(r['cx'])} | {fmt(r['cy'])} | {fmt(r['focal_px'])} | {fmt(r['pp_error_px'])} | {fmt(r['objective'], 6)} |")
    (table_dir / "joint_focal_n24_per_scene.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if colmap_seq is not None or colmap_all is not None:
        seq_rows = colmap_seq or []
        all_rows = colmap_all or []
        ours = {r["scene"]: float(r["pp_error_px"]) for r in n24}
        lines = ["# PCCC main comparison", "", "| baseline | scenes | COLMAP mean pp | COLMAP median pp | ours mean pp | ours median pp | ours wins | COLMAP wins |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for label, rows in [("COLMAP known-RT seq", seq_rows), ("COLMAP known-RT all", all_rows)]:
            if not rows:
                continue
            col_pp = [float(r["colmap"]["intrinsics"].get("principal_error_px", math.nan)) for r in rows]
            ours_pp = [float(r.get("ours_joint_pp_error_px", math.nan)) for r in rows]
            col_s = stats(col_pp)
            ours_s = stats(ours_pp)
            paired = [(c, o) for c, o in zip(col_pp, ours_pp) if math.isfinite(c) and math.isfinite(o)]
            col_wins = sum(1 for c, o in paired if c < o)
            lines.append(f"| {label} | {len(rows)} | {fmt(col_s['mean'])} | {fmt(col_s['median'])} | {fmt(ours_s['mean'])} | {fmt(ours_s['median'])} | {len(paired) - col_wins} | {col_wins} |")
        (table_dir / "main_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

        lines = ["# PCCC COLMAP per-scene comparison", "", "| scene | COLMAP seq pp | COLMAP all pp | ours pp |", "|---|---:|---:|---:|"]
        seq_by_scene = {r["scene"]: r for r in seq_rows}
        all_by_scene = {r["scene"]: r for r in all_rows}
        for scene in SCENES:
            if scene not in ours:
                continue
            seq_pp = seq_by_scene.get(scene, {}).get("colmap", {}).get("intrinsics", {}).get("principal_error_px", math.nan)
            all_pp = all_by_scene.get(scene, {}).get("colmap", {}).get("intrinsics", {}).get("principal_error_px", math.nan)
            lines.append(f"| {scene} | {fmt(seq_pp)} | {fmt(all_pp)} | {fmt(ours[scene])} |")
        (table_dir / "colmap_per_scene.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone PCCC reproduction: data, COLMAP, joint focal, and tables.")
    parser.add_argument("--work-dir", type=Path, default=WORK)
    parser.add_argument("--source-tanks-root", type=Path, default=None)
    parser.add_argument("--colmap-bin", default=None)
    parser.add_argument("--scenes", nargs="*", default=SCENES)
    parser.add_argument("--counts", type=int, nargs="+", default=[6, 12, 24])
    parser.add_argument("--crop-size", type=int, default=480)
    parser.add_argument("--max-features", type=int, default=6000)
    parser.add_argument("--ratio-test", type=float, default=0.75)
    parser.add_argument("--trim-quantile", type=float, default=0.9)
    parser.add_argument("--skip-colmap", action="store_true")
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    work_dir = args.work_dir.resolve()
    if args.clean and work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    tanks_root = ensure_tanks_root(source_tanks_root=args.source_tanks_root, work_dir=work_dir)
    manifest_path = prepare_crops_and_manifest(tanks_root=tanks_root, work_dir=work_dir, scenes=list(args.scenes), counts=list(args.counts), crop_size=int(args.crop_size))
    colmap_bin = None if args.skip_colmap else ensure_colmap(raw=args.colmap_bin, work_dir=work_dir)

    results_root = work_dir / "results"
    joint_rows = run_joint_focal(
        manifest_path=manifest_path,
        output_dir=results_root / "joint_focal",
        scenes=list(args.scenes),
        counts=list(args.counts),
        max_features=int(args.max_features),
        ratio_test=float(args.ratio_test),
        trim_quantile=float(args.trim_quantile),
    )

    colmap_seq = None
    colmap_all = None
    if colmap_bin is not None:
        colmap_counts = [count for count in args.counts if int(count) == 24]
        if not colmap_counts:
            colmap_counts = [max(args.counts)]
        colmap_seq = run_colmap_baseline(
            manifest_path=manifest_path,
            output_dir=results_root / "colmap_known_rt_seq",
            colmap_bin=colmap_bin,
            policy="seq",
            scenes=list(args.scenes),
            counts=colmap_counts,
            joint_rows=joint_rows,
            max_features=int(args.max_features),
            ratio_test=float(args.ratio_test),
        )
        colmap_all = run_colmap_baseline(
            manifest_path=manifest_path,
            output_dir=results_root / "colmap_known_rt_all",
            colmap_bin=colmap_bin,
            policy="all",
            scenes=list(args.scenes),
            counts=colmap_counts,
            joint_rows=joint_rows,
            max_features=int(args.max_features),
            ratio_test=float(args.ratio_test),
        )

    write_tables(output_dir=results_root, joint_rows=joint_rows, colmap_seq=colmap_seq, colmap_all=colmap_all)
    info(f"WROTE {results_root / 'tables'}")


if __name__ == "__main__":
    main()

