python zero_shot_eval.py \
  --model Maple728/TimeMoE-50M \
  --input_dir ./test_fold/ \
    --settings "512:96,1024:192" \
    --batch_size 16 \
    --gating_png moe_all.png \
    --cmap coolwarm \
    --vmax 0.4           # 可按需要调