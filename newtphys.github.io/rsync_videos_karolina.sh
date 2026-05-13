PREFIX=/scratch/project/eu-25-92/composite_physics/dataset/simulation_v4
DEST=./downloaded

LIST_FOLDERS=(
    dl3dv/yms-variations/soft/4/c-1_no-4_d-10_s-dl3dv-all_models-hf-gso_MLP-10_smooth_h-10-40_seed-29_20251214_063652
    dl3dv/random/5/c-1_no-5_d-10_s-dl3dv-all_models-hf-gso_MLP-10_smooth_h-10-40_seed-125_20251212_191943
    dl3dv/yms-variations/soft/6/c-1_no-6_d-10_s-dl3dv-all_models-hf-gso_MLP-10_smooth_h-10-40_seed-14_20251214_160155
    dl3dv/random/3/c-1_no-3_d-10_s-dl3dv-all_models-hf-gso_MLP-10_smooth_h-10-40_seed-5_20251212_034543
    dl3dv/random/6/c-1_no-6_d-10_s-dl3dv-all_models-hf-gso_MLP-10_smooth_h-10-40_seed-174_20251213_072253
    dl3dv/random/6/c-1_no-6_d-10_s-dl3dv-all_models-hf-gso_MLP-10_smooth_h-10-40_seed-181_20251213_073854
)

mkdir -p "$DEST"

tmpfile=$(mktemp)
printf '%s/\n' "${LIST_FOLDERS[@]}" | sort > "$tmpfile"

rsync -avr --progress \
    -e 'ssh -i ~/.ssh/id_rsa_karolina' \
    --files-from="$tmpfile" \
    --exclude='render_circling_no_text_layout_position/' \
    --exclude='render_roi_circled/' \
    --exclude='render_circling_no_text_no_layout_position/' \
    --exclude='render_circling_text_layout_position/' \
    --exclude='render_circling_text_no_layout_position/' \
    "it4i-thvu@login2.karolina.it4i.cz:${PREFIX}/" \
    "$DEST/"

rm -f "$tmpfile"