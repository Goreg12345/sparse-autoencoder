# name mover outputs
python store_activations_to_disc.py --store_size=61440000 --actv_name="blocks.9.attn.hook_z" --layer=9 --head=9 --gpu_num=3
python store_activations_to_disc.py --store_size=61440000 --actv_name="blocks.9.attn.hook_z" --layer=9 --head=6 --gpu_num=3
python store_activations_to_disc.py --store_size=61440000 --actv_name="blocks.10.attn.hook_z" --layer=10 --head=0 --gpu_num=3
# name mover queries
python store_activations_to_disc.py --store_size=61440000 --actv_name="blocks.9.attn.hook_q" --layer=9 --head=9 --gpu_num=3
python store_activations_to_disc.py --store_size=61440000 --actv_name="blocks.9.attn.hook_q" --layer=9 --head=6 --gpu_num=3
python store_activations_to_disc.py --store_size=61440000 --actv_name="blocks.10.attn.hook_q" --layer=10 --head=0 --gpu_num=3
# si values
python store_activations_to_disc.py --store_size=61440000 --actv_name="blocks.8.attn.hook_v" --layer=8 --head=6 --gpu_num=3
python store_activations_to_disc.py --store_size=61440000 --actv_name="blocks.8.attn.hook_v" --layer=8 --head=10 --gpu_num=3
