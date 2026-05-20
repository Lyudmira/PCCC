# PCCC minimal reproduction

This directory is a standalone reproduction package for the PCCC paper data.
It does not import code from `KFPPS/src`, does not read the existing KFPPS
manifest, and does not read existing KFPPS result files.

One command on Windows:

```powershell
.\run.ps1
```

Equivalent Python command:

```powershell
python .\reproduce_pccc.py
```

What it does:

1. Downloads the InstantSplat Tanks dataset from Hugging Face, unless a local
   Tanks root is provided.
2. Prepares left-top `480 x 480` crops for the 8 paper scenes.
3. Downloads a Windows COLMAP release from the official COLMAP GitHub release,
   unless `COLMAP_BIN` or `--colmap-bin` is provided.
4. Runs the paper mainline: pose-conditioned joint focal recovery.
5. Runs the paper COLMAP known-RT baselines for `seq` and `all` pair graphs.
6. Runs the more-information COLMAP controls: `all + oracle F` and
   `all + oracle F inliers`, both still under the same known-RT protocol.
7. Writes paper tables under `work/results/tables`.

Useful shorter commands:

```powershell
# Use an existing local Tanks directory instead of downloading it.
python .\reproduce_pccc.py --source-tanks-root C:\path\to\InstantSplat\Tanks

# Only run the paper method, no COLMAP baseline.
python .\reproduce_pccc.py --skip-colmap

# Skip only the extra oracle-F COLMAP controls.
python .\reproduce_pccc.py --skip-colmap-more-info

# Smoke test one scene.
python .\reproduce_pccc.py --scenes Ballroom --counts 24 --skip-colmap

# Optional development-only N sweep.
python .\reproduce_pccc.py --counts 6 12 24 --skip-colmap
```

Outputs:

- `work/data/source/InstantSplat/Tanks`: raw dataset.
- `work/data/crops`: generated crop images.
- `work/external/colmap`: downloaded COLMAP, if needed.
- `work/results/joint_focal/results.json`: paper method results.
- `work/results/colmap_known_rt_seq/results.json`: COLMAP seq baseline.
- `work/results/colmap_known_rt_all/results.json`: COLMAP all-pair baseline.
- `work/results/colmap_known_rt_all_oracle_f/results.json`: COLMAP all-pair
  baseline with exact-pose/GT-intrinsics pairwise geometry written into
  `two_view_geometries`.
- `work/results/colmap_known_rt_all_oracle_f_inliers/results.json`: the same,
  with matches filtered by oracle Sampson residual before COLMAP triangulation.
- `work/results/tables/*.md`: paper-ready tables.
