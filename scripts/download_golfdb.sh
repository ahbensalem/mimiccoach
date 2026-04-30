#!/usr/bin/env bash
# Pull a curated GolfDB subset to backend/reference/data/golfdb/.
#
# What it pulls:
#   1. golfDB.mat — metadata (1,400 annotations across 580 YouTube videos),
#      directly from the GolfDB GitHub repo.
#   2. pose_landmarker_full.task — MediaPipe Pose Landmarker model bundle
#      (the same one the Modal image bakes at /opt/pose_landmarker.task).
#   3. ~10 curated YouTube source videos picked for diversity across
#      sex × club × view × player. Subsequent annotations are cropped +
#      sliced in-loader.
#
# The official GolfDB videos_160.zip artifact (cropped 160×160 clips for
# all 1,400 annotations) is 18 GB on Google Drive — too big to wire up
# to a one-shot script. Pulling source videos for a curated subset gives
# us the same downstream pipeline output for ~50–80 clips at a fraction
# of the bandwidth.
#
# Re-run-safe: every download is skipped if the target file already
# exists.

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="backend/reference/data/golfdb"
mkdir -p "$ROOT/videos" "$ROOT/pose_cache"

echo "==> Metadata: golfDB.mat"
if [[ ! -f "$ROOT/golfDB.mat" ]]; then
  curl -fsSL "https://raw.githubusercontent.com/wmcnally/golfdb/master/data/golfDB.mat" \
    -o "$ROOT/golfDB.mat"
else
  echo "    already present"
fi

echo "==> MediaPipe pose model"
MODEL_PATH="$ROOT/pose_landmarker_full.task"
if [[ ! -f "$MODEL_PATH" ]]; then
  curl -fsSL \
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task" \
    -o "$MODEL_PATH"
else
  echo "    already present"
fi

# Diverse curated subset. Each entry: <youtube_id>  <player / club / view>
# Picked for: m+f, multiple clubs, both views, top annotation counts.
CURATED_VIDEOS=(
  "bMVf3lKrUlo"  # Bernhard Langer  · m / iron / face-on        · 15 anno
  "XIzyJGDBuqc"  # Fred Couples     · m / driver / face-on      · 14 anno
  "skkF2JsuLWM"  # Bubba Watson     · m / iron / down-the-line  ·  8 anno
  "ghi3184d6Is"  # Lydia Ko         · f / driver / down-the-line·  8 anno
  "IVRbQrq2JHo"  # Michelle Wie     · f / fairway / face-on     · 16 anno
  "3zZXUQywgSw"  # Brooke Henderson · f / wedge / down-the-line ·  5 anno
  "H9MWcs3YS-I"  # Tiger Woods      · m / iron / face-on        ·  5 anno
  "YuEATi-0HMM"  # Rory McIlroy     · m / driver / down-the-line·  4 anno
)

echo "==> YouTube source videos (${#CURATED_VIDEOS[@]} clips)"
ABS_ROOT="$PWD/$ROOT"
pushd backend >/dev/null
for yt in "${CURATED_VIDEOS[@]}"; do
  out="$ABS_ROOT/videos/${yt}.mp4"
  if [[ -f "$out" ]]; then
    echo "    [skip] $yt — already on disk"
    continue
  fi
  echo "    [pull] $yt"
  if uv run --quiet --group ingest yt-dlp \
      --quiet --no-warnings --no-playlist \
      -f "best[ext=mp4][height<=720]/best[ext=mp4]/best" \
      --merge-output-format mp4 \
      -o "$out" \
      "https://www.youtube.com/watch?v=${yt}"
  then
    :
  else
    echo "    [fail] $yt — skipping (video may be unavailable)"
    rm -f "$out"
  fi
done
popd >/dev/null

echo "==> Done. Wired data lives in $ROOT/"
echo "    Run scripts/build_library.sh next to rebuild the manifest."
