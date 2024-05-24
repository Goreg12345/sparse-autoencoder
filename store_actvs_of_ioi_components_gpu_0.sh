# duplicate token heads
python store_activations_to_disc.py --store_size=61440000 --actv_name="blocks.0.attn.hook_z" --layer=0 --head=1 --gpu_num=0
python store_activations_to_disc.py --store_size=61440000 --actv_name="blocks.3.attn.hook_z" --layer=3 --head=0 --gpu_num=0
python store_activations_to_disc.py --store_size=61440000 --actv_name="blocks.0.attn.hook_z" --layer=0 --head=10 --gpu_num=0
# induction heads
python store_activations_to_disc.py --store_size=61440000 --actv_name="blocks.5.attn.hook_z" --layer=5 --head=5 --gpu_num=0
python store_activations_to_disc.py --store_size=61440000 --actv_name="blocks.6.attn.hook_z" --layer=6 --head=9 --gpu_num=0
python store_activations_to_disc.py --store_size=61440000 --actv_name="blocks.5.attn.hook_z" --layer=5 --head=8 --gpu_num=0
python store_activations_to_disc.py --store_size=61440000 --actv_name="blocks.5.attn.hook_z" --layer=5 --head=9 --gpu_num=0
# S-inhibition heads out
python store_activations_to_disc.py --store_size=61440000 --actv_name="blocks.7.attn.hook_z" --layer=7 --head=3 --gpu_num=0
python store_activations_to_disc.py --store_size=61440000 --actv_name="blocks.7.attn.hook_z" --layer=7 --head=9 --gpu_num=0
python store_activations_to_disc.py --store_size=61440000 --actv_name="blocks.8.attn.hook_z" --layer=8 --head=6 --gpu_num=0
python store_activations_to_disc.py --store_size=61440000 --actv_name="blocks.8.attn.hook_z" --layer=8 --head=10 --gpu_num=0
