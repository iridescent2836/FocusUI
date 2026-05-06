#!/usr/bin/env bash
# =============================================================================
# Download raw images from GUI-Actor-Data and place them where data_config.yaml
# expects them (under datasets/GUI-Actor-Data/).
#
# Run from the project root:
#   bash download_gui_actor_data.sh
#
# You can skip individual datasets by commenting out the corresponding block.
# Total sizes:
#   GUIAct        ~4 GB
#   GUIEnv        ~6 GB
#   Wave-UI       ~24 GB
#   AndroidControl ~49 GB
#   AMEX          ~92 GB  (3-part split)
#   UGround       ~214 GB (6-part split) ← largest, skip if disk is tight
# =============================================================================

set -euo pipefail

HF_REPO="cckevinn/GUI-Actor-Data"
DEST="datasets/GUI-Actor-Data"
TMP="${DEST}/_tmp_zips"

mkdir -p "$TMP"

# Helper: download a single file from the HF dataset repo
hf_dl() {
    local filename="$1"
    huggingface-cli download "$HF_REPO" \
        --repo-type dataset \
        --include "$filename" \
        --local-dir "$TMP"
}

# =============================================================================
# 1. GUIAct  →  datasets/GUI-Actor-Data/GUIAct/web_imgs/
# =============================================================================
echo "=== Downloading GUIAct (~4 GB) ==="
hf_dl "GUIAct_images.zip"
unzip -q "$TMP/GUIAct_images.zip" -d "$DEST"
echo "GUIAct done."

# =============================================================================
# 2. GUIEnv  →  datasets/GUI-Actor-Data/guienvs/images/
# =============================================================================
# echo "=== Downloading GUIEnv (~6 GB) ==="
# hf_dl "GUIEnv_images.zip"
# unzip -q "$TMP/GUIEnv_images.zip" -d "$DEST"
# echo "GUIEnv done."

# =============================================================================
# 3. Wave-UI  →  datasets/GUI-Actor-Data/Wave-UI/images_fixed/
# =============================================================================
# echo "=== Downloading Wave-UI (~24 GB) ==="
# hf_dl "Wave-UI_images.zip"
# unzip -q "$TMP/Wave-UI_images.zip" -d "$DEST"
# echo "Wave-UI done."

# =============================================================================
# 4. AndroidControl  →  datasets/GUI-Actor-Data/AndroidControl/tfrecord/images/
# =============================================================================
# echo "=== Downloading AndroidControl (~49 GB) ==="
# hf_dl "AndroidControl_images.zip"
# unzip -q "$TMP/AndroidControl_images.zip" -d "$DEST"
# echo "AndroidControl done."

# =============================================================================
# 5. AMEX  →  datasets/GUI-Actor-Data/AMEX/screenshot/
#    Three-part split: cat the parts together first, then unzip.
# =============================================================================
# echo "=== Downloading AMEX (~92 GB, 3 parts) ==="
# hf_dl "amex_images_part_aa"
# hf_dl "amex_images_part_ab"
# hf_dl "amex_images_part_ac"
# echo "Joining AMEX parts..."
# cat "$TMP/amex_images_part_aa" \
#     "$TMP/amex_images_part_ab" \
#     "$TMP/amex_images_part_ac" > "$TMP/amex_images.zip"
# unzip -q "$TMP/amex_images.zip" -d "$DEST"
# rm -f "$TMP/amex_images_part_"* "$TMP/amex_images.zip"
# echo "AMEX done."

# =============================================================================
# 6. UGround  →  datasets/GUI-Actor-Data/UGround-V1-Data-Box/images/
#    Six-part 7-zip split: .z01-.z05 + .zip  (requires p7zip-full)
#    ~214 GB — comment out this block if disk space is limited.
# =============================================================================
# echo "=== Downloading UGround (~214 GB, 6 parts) ==="
# for ext in z01 z02 z03 z04 z05 zip; do
#     hf_dl "Uground_images_split.${ext}"
# done
# echo "Extracting UGround (7z)..."
# # 7z automatically finds the other parts when given the .zip piece
# 7z x "$TMP/Uground_images_split.zip" -aoa -o"$DEST"
# rm -f "$TMP/Uground_images_split".*
# echo "UGround done."

# =============================================================================
# Clean up temp zips
# =============================================================================
rm -rf "$TMP"

# =============================================================================
# Verify extracted paths match data_config.yaml expectations
# =============================================================================
echo ""
echo "=== Verifying extracted paths ==="
declare -A EXPECTED_PATHS=(
    ["GUIAct"]="$DEST/GUIAct/web_imgs"
    # ["GUIEnv"]="$DEST/guienvs/images"
    # ["Wave-UI"]="$DEST/Wave-UI/images_fixed"
    # ["AndroidControl"]="$DEST/AndroidControl/tfrecord/images"
    # ["AMEX"]="$DEST/AMEX/screenshot"
    # ["UGround"]="$DEST/UGround-V1-Data-Box/images"
)

all_ok=true
for name in "${!EXPECTED_PATHS[@]}"; do
    path="${EXPECTED_PATHS[$name]}"
    if [ -d "$path" ]; then
        count=$(find "$path" -maxdepth 1 -type f | wc -l)
        echo "  [OK]  $name  →  $path  ($count files)"
    else
        echo "  [MISSING]  $name  →  $path"
        echo "             The zip may have extracted to a different subfolder."
        echo "             Run: ls $DEST/ to check, then create a symlink or rename."
        all_ok=false
    fi
done

if $all_ok; then
    echo ""
    echo "All paths OK — you can now run training with:"
    echo "  bash stage_1_ft_focusui_scorer.sh"
    echo "  (or: python train_patch_scorer.py --data_path data/data_config.yaml ...)"
else
    echo ""
    echo "Some paths are missing. See fix_paths.sh for how to create symlinks."
fi
