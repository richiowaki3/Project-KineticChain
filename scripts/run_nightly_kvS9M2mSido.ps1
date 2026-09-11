# -*- coding: utf-8 -*-
# Nightly execution script for 4D-Humans & Laban Kinematics analysis
# Target: kvS9M2mSido (Popping / Alien Invasion Dance)

$ErrorActionPreference = "Continue"
$BASE = "D:\motion_capture"
$B    = "D:/motion_capture"
$SMPL = "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"
$id   = "kvS9M2mSido"
$log  = "$BASE\night_run_$id.log"

function Log($msg) {
    $time = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$time] $msg"
    Write-Host $line
    Add-Content -Path $log -Value $line -Encoding UTF8
}

Log "========================================================"
Log "Nightly Batch Job Triggered: kvS9M2mSido (4D-Humans)"
Log "========================================================"

# Step 1: Verify Input Video
$videoPath = "$BASE\input_videos\$id.mp4"
if (-not (Test-Path $videoPath)) {
    Log "[ERROR] Input video missing: $videoPath"
    exit 1
}
Log "[OK] Input video ready: $videoPath"

# Step 2: Ensure Directories
New-Item -ItemType Directory -Force "$BASE\output\4dhumans_$id" | Out-Null
New-Item -ItemType Directory -Force "$BASE\output_results" | Out-Null

# Step 3: Run 4D-Humans Docker
$prep = "cp /data/models/$SMPL /app/4D-Humans/ && mkdir -p /app/4D-Humans/data && cp /data/models/$SMPL /app/4D-Humans/data/ && python track.py video.source=/data/input_videos/$id.mp4 video.start_frame=0 video.end_frame=999999"
Log "[START] Running 4D-Humans GPU inference (full video)..."

docker run --gpus all --rm `
  -v "${B}:/data" `
  -v "${B}/output/4dhumans_${id}:/app/4D-Humans/outputs" `
  -v "${B}/.phalp_cache:/root/.cache/phalp" `
  -v "${B}/.4dhumans_cache:/root/.cache/4DHumans" `
  -v "${B}/.torch_cache:/root/.cache/torch" `
  -e PYOPENGL_PLATFORM=osmesa `
  -e NVIDIA_DRIVER_CAPABILITIES=all `
  -e HYDRA_FULL_ERROR=1 `
  4dhumans:latest bash -c $prep *>> $log

Log "[DONE] Docker 4D-Humans run completed."

# Step 4: Copy Results to output_results
$previewSrc = Get-ChildItem "$BASE\output\4dhumans_$id\*.mp4" | Select-Object -First 1
$pklSrc = "$BASE\output\4dhumans_$id\results\demo_$id.pkl"

if ($previewSrc) {
    Copy-Item $previewSrc.FullName "$BASE\output_results\$($id)__4dhumans_preview.mp4" -Force
    Log "[OK] Preview MP4 saved to $BASE\output_results\$($id)__4dhumans_preview.mp4"
} else {
    Log "[WARN] Preview MP4 not found in outputs"
}

if (Test-Path $pklSrc) {
    Copy-Item $pklSrc "$BASE\output_results\$($id)__4dhumans_tracks.pkl" -Force
    Log "[OK] PKL tracks saved to $BASE\output_results\$($id)__4dhumans_tracks.pkl"
} else {
    Log "[ERROR] Output PKL tracks missing: $pklSrc"
    exit 1
}

# Step 5: Kinematics Batch Pipeline & Laban Library Indexer
Log "[START] Processing DanceKinematics batch pipeline..."
Set-Location "D:\Antigravity_Work\MotionAnalysis"

python examples\run_batch_all.py *>> $log
Log "[OK] Batch kinematics pipeline finished."

Log "[START] Building Laban Effort library bundle..."
python src\storage\laban_library_indexer.py *>> $log
Log "[OK] Laban library bundle updated."

Log "[START] Generating sample composed dance..."
python examples\demo_assemble_laban.py *>> $log
Log "[OK] Sample composed dance generated."

Log "========================================================"
Log "ALL NIGHTLY TASKS SUCCESSFULLY COMPLETED!"
Log "========================================================"
