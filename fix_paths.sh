#!/usr/bin/env bash
# =============================================================================
# If download_gui_actor_data.sh reports MISSING paths, run this script to
# inspect what actually got extracted and create symlinks to fix the mapping.
#
# Run from the project root:
#   bash fix_paths.sh
# =============================================================================

DEST="datasets/GUI-Actor-Data"

echo "Contents of $DEST/:"
ls "$DEST/"
echo ""

# For each expected path, print what actually exists so you can identify
# the correct source name to symlink from.
declare -A EXPECTED_PATHS=(
    ["GUIAct"]="$DEST/GUIAct/web_imgs"
    ["GUIEnv"]="$DEST/guienvs/images"
    ["Wave-UI"]="$DEST/Wave-UI/images_fixed"
    ["AndroidControl"]="$DEST/AndroidControl/tfrecord/images"
    ["AMEX"]="$DEST/AMEX/screenshot"
    ["UGround"]="$DEST/UGround-V1-Data-Box/images"
)

for name in "${!EXPECTED_PATHS[@]}"; do
    expected="${EXPECTED_PATHS[$name]}"
    if [ ! -d "$expected" ]; then
        echo "[MISSING] $name expected at: $expected"
        echo "  Possible matches under $DEST/:"
        find "$DEST" -maxdepth 3 -type d | grep -i "${name,,}" || echo "  (none found)"
        echo ""
    fi
done

echo "To create a symlink, run e.g.:"
echo "  ln -s $(pwd)/datasets/GUI-Actor-Data/ACTUAL_PATH $(pwd)/datasets/GUI-Actor-Data/EXPECTED_PATH"
