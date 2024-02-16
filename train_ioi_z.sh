python train.py --gpu_num=0 --actv_name="blocks.0.attn.hook_z" --layer=0 --head=1 --l1_coefficient=0.006
python train.py --gpu_num=0 --actv_name="blocks.3.attn.hook_z" --layer=3 --head=0 --l1_coefficient=0.006
python train.py --gpu_num=0 --actv_name="blocks.0.attn.hook_z" --layer=0 --head=10 --l1_coefficient=0.006
python train.py --gpu_num=0 --actv_name="blocks.7.attn.hook_z" --layer=7 --head=3 --l1_coefficient=0.006
python train.py --gpu_num=0 --actv_name="blocks.7.attn.hook_z" --layer=7 --head=9 --l1_coefficient=0.006
python train.py --gpu_num=0 --actv_name="blocks.8.attn.hook_z" --layer=8 --head=6 --l1_coefficient=0.006
python train.py --gpu_num=0 --actv_name="blocks.8.attn.hook_z" --layer=8 --head=10 --l1_coefficient=0.006