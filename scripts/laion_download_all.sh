#!/bin/sh

if hash wandb 2> /dev/null; then
    wandb login
    ENABLE_WANDB=True
else
    ENABLE_WANDB=False
fi

# Use half the CPU cores so cloudwriter can use other half
img2dataset \
    --url_list /tmp/laion2b-raw \
    --input_format parquet \
    --url_col url \
    --caption_col caption \
    --output_format parquet \
    --output_folder /tmp/laion2b-processed \
    --processes_count 32 \
    --thread_count 64 \
    --resize_mode no \
    --compute_hash "md5" \
    --verify_hash '["md5","md5"]' \
    --save_additional_columns '["punsafe","pwatermark","similarity","hash"]' \
    --enable_wandb True \
    --wandb_project laion-dataset

touch /tmp/laion2b-processed/done
