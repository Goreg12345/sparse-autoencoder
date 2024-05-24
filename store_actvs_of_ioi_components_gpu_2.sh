# name mover values
python store_activations_to_disc.py --store_size=61440000 --actv_name="blocks.9.attn.hook_v" --layer=9 --head=9 --gpu_num=2
python store_activations_to_disc.py --store_size=61440000 --actv_name="blocks.9.attn.hook_v" --layer=9 --head=6 --gpu_num=2
python store_activations_to_disc.py --store_size=61440000 --actv_name="blocks.10.attn.hook_v" --layer=10 --head=0 --gpu_num=2
# name mover keys
python store_activations_to_disc.py --store_size=61440000 --actv_name="blocks.9.attn.hook_k" --layer=9 --head=9 --gpu_num=2
python store_activations_to_disc.py --store_size=61440000 --actv_name="blocks.9.attn.hook_k" --layer=9 --head=6 --gpu_num=2
python store_activations_to_disc.py --store_size=61440000 --actv_name="blocks.10.attn.hook_k" --layer=10 --head=0 --gpu_num=2
# S-inhibition heads values
python store_activations_to_disc.py --store_size=61440000 --actv_name="blocks.7.attn.hook_v" --layer=7 --head=3 --gpu_num=2
python store_activations_to_disc.py --store_size=61440000 --actv_name="blocks.7.attn.hook_v" --layer=7 --head=9 --gpu_num=2
