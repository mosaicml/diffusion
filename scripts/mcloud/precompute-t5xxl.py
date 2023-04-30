import textwrap

from mcli.sdk import create_run, RunConfig

cluster = 'r1z1'
gpu_num = 8
image = 'mosaicml/pytorch_vision:1.13.1_cu117-python3.10-ubuntu20.04'
integrations = [
    {
        'integration_type': 'git_repo',
        'git_repo': 'mosaicml/diffusion',
        'git_branch': 'mvpatel2000/t5-precompute',
    },
    {
        'integration_type': 'wandb',
        'project': 'precompute-latents-t5xxl',
        'entity': 'mosaic-ml',
    },
]


folders = [
    '64-128',
    '128-256',
    '256-512',
    '512-768',
    '768-1024',
    '1024-1048576',
]

for folder in folders:
    for bucket in range(1, 5):

        run_name = f'precompute-t5xxl-{folder}-{bucket}'

        command = textwrap.dedent(f"""
        cd diffusion
        pip install -e .
        pip install protobuf==3.20.0
        pip install bs4
        pip install ftfy
        cd scripts

        composer precompute_t5xxl.py \
        --remote_download oci://mosaicml-internal-dataset-laion2b-en/4.5v2/{folder}/ \
        --local /tmp/mds-cache/mds-laion2b-en/4.5/{folder}/ \
        --remote_upload oci://mosaicml-internal-dataset-laion2b-en/4.5v2-t5xxl/{folder}/  \
        --bucket {bucket} \
        --wandb_name {folder}-bucket-{bucket}
        """)

        print(f'Launching run {run_name}')
        run_config = RunConfig(
            run_name=run_name,
            cluster=cluster,
            gpu_num=gpu_num,
            image=image,
            integrations=integrations,
            command=command,
        )
        create_run(run_config)