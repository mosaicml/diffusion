#!/bin/sh

if hash wandb 2> /dev/null; then
    wandb login
    ENABLE_WANDB=True
else
    ENABLE_WANDB=False
fi

# Use half the CPU cores so cloudwriter can use other half
img2dataset \
    --url_list /tmp/laion2b-4.5-raw \
    --input_format parquet \
    --url_col URL \
    --caption_col TEXT \
    --output_format parquet \
    --output_folder /tmp/laion2b-processed \
    --processes_count 32 \
    --thread_count 64 \
    --resize_mode no \
    --save_additional_columns '["punsafe","pwatermark","similarity","hash","AESTHETIC_SCORE"]' \
    --enable_wandb True \
    --wandb_project laion-dataset

touch /tmp/laion2b-processed/done
