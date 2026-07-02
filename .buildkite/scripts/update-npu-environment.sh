#!/bin/bash
# Purpose: Updates NPU test environment to match PR changes
#          - Saves and reverts all old patches before code update (for proper revert)
#          - Updates VIME code to the specified commit
#          - Applies all new patches to corresponding components
#          - Sorts ASCEND_VISIBLE_DEVICES for consistent device ordering
# Usage: Called by Buildkite pipeline during NPU test runs
# set -e

VIME_DIR="/root/vime"

declare -A PATCH_CONFIGS=(
    ["vllm.patch"]="/root/vllm"
    ["vllm-ascend.patch"]="/root/vllm-ascend"
    ["megatron_comm.patch"]="/root/Megatron-LM"
    ["megatron.patch"]="/root/Megatron-LM"
    ["megatron-bridge.patch"]="/root/Megatron-Bridge"
    ["mindspeed.patch"]="/root/MindSpeed"
)

update_vime_code() {
    echo "INFO: Updating VIME code..."
    cd "$VIME_DIR"

    if [ -n "${BUILDKITE_COMMIT}" ]; then
        echo "INFO: Fetching and checking out commit ${BUILDKITE_COMMIT}"
        git fetch origin "${BUILDKITE_COMMIT}"
        git checkout "${BUILDKITE_COMMIT}"
        pip install -e . --no-deps --break-system-packages || pip install -e . --no-deps
    else
        echo "INFO: BUILDKITE_COMMIT not set, skipping code update"
    fi
}

sort_ascend_visible_devices() {
    if [ -n "${ASCEND_VISIBLE_DEVICES}" ]; then
        SORTED_DEVICES=$(echo "${ASCEND_VISIBLE_DEVICES}" | tr ',' '\n' | sort -n | tr '\n' ',')
        SORTED_DEVICES=${SORTED_DEVICES%,}
        export ASCEND_VISIBLE_DEVICES=$SORTED_DEVICES
        echo "Sorted ASCEND_VISIBLE_DEVICES: $ASCEND_VISIBLE_DEVICES"
    fi
}

get_patch_component() {
    local patch_name="$1"
    local config="${PATCH_CONFIGS[$patch_name]}"
    echo "$config"
}

is_patch_applied() {
    local component_dir="$1"
    local patch_path="$2"

    if git -C "$component_dir" apply --reverse --check --whitespace=nowarn "$patch_path"; then
        return 0
    else
        return 1
    fi
}

revert_patch() {
    local component_dir="$1"
    local patch_name="$2"
    local old_patch_path="$3"

    if [ -f "$old_patch_path" ]; then
        echo "INFO: Attempting to reverse-apply old patch from $old_patch_path"
        if git -C "$component_dir" apply --reverse --whitespace=nowarn "$old_patch_path"; then
            echo "INFO: Successfully reverted old $patch_name"
        else
            echo "WARNING: Failed to reverse-apply old patch $patch_name, skipping"
        fi
    else
        echo "INFO: Old patch $patch_name not found at $old_patch_path, skipping revert"
    fi
}

apply_patch() {
    local component_dir="$1"
    local patch_path="$2"
    local patch_name="$3"

    echo "INFO: Applying $patch_name to $component_dir"
    if git -C "$component_dir" apply --whitespace=nowarn "$patch_path"; then
        echo "INFO: Successfully applied $patch_name"
    else
        echo "ERROR: Failed to apply $patch_name to $component_dir"
        sleep 3600
        exit 1
    fi
}

save_old_patches() {
    local backup_dir="${VIME_DIR}/.old_patches"

    echo "INFO: Saving old patches to $backup_dir..."
    mkdir -p "$backup_dir"

    for patch_name in "${!PATCH_CONFIGS[@]}"; do
        local patch_path="${VIME_DIR}/docker/npu_patch/$patch_name"
        if [ -f "$patch_path" ]; then
            cp "$patch_path" "$backup_dir/$patch_name"
            echo "INFO: Saved old $patch_name"
        else
            echo "WARNING: Patch file $patch_path not found, skipping backup"
        fi
    done

    echo "$backup_dir"
}

revert_all_patches() {
    local old_patch_dir="$1"

    echo "INFO: Reverting all already applied patches..."

    for patch_name in "${!PATCH_CONFIGS[@]}"; do
        local component_dir=$(get_patch_component "$patch_name")
        local old_patch_path="$old_patch_dir/$patch_name"

        if is_patch_applied "$component_dir" "$old_patch_path"; then
            echo "INFO: $patch_name is currently applied, reverting..."
            revert_patch "$component_dir" "$patch_name" "$old_patch_path"
        else
            echo "INFO: $patch_name is not currently applied, nothing to revert"
        fi
    done
}

apply_all_patches() {
    echo "INFO: Applying all patches..."

    for patch_name in "${!PATCH_CONFIGS[@]}"; do
        local component_dir=$(get_patch_component "$patch_name")
        local patch_path="${VIME_DIR}/docker/npu_patch/$patch_name"

        if [ -f "$patch_path" ]; then
            apply_patch "$component_dir" "$patch_path" "$patch_name"
        else
            echo "WARNING: Patch file $patch_path not found, skipping"
        fi
    done
}

main() {
    echo "=== Step 1: Save all old patches before code update ==="
    local old_patch_dir=$(save_old_patches)

    echo "=== Step 2: Update VIME code ==="
    update_vime_code

    echo "=== Step 3: Revert all already applied patches ==="
    revert_all_patches "$old_patch_dir"

    echo "=== Step 4: Apply all patches ==="
    apply_all_patches

    echo "INFO: NPU environment update completed successfully"

    echo "=== Step 5: Sort ASCEND_VISIBLE_DEVICES ==="
    sort_ascend_visible_devices
}

main