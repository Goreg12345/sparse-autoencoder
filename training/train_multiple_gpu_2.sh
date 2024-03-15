# python train.py --gpu_num=2 --actv_name="blocks.7.attn.hook_z" --layer=7 --head=3 --l1_coefficient=0.015 --allow_lower_decoder_norm=True --batch_size=512 --d_hidden=1000 --start_lr_decay=440000 --train_steps=488282
# python train.py --gpu_num=2 --actv_name="blocks.7.attn.hook_z" --layer=7 --head=9 --l1_coefficient=0.015 --allow_lower_decoder_norm=True --batch_size=512 --d_hidden=1000 --start_lr_decay=440000 --train_steps=488282
# python train.py --gpu_num=2 --actv_name="blocks.8.attn.hook_z" --layer=8 --head=6 --l1_coefficient=0.015 --allow_lower_decoder_norm=True --batch_size=512 --d_hidden=1000 --start_lr_decay=440000 --train_steps=488282
# python train.py --gpu_num=2 --actv_name="blocks.8.attn.hook_z" --layer=8 --head=10 --l1_coefficient=0.015 --allow_lower_decoder_norm=True --batch_size=512 --d_hidden=1000 --start_lr_decay=440000 --train_steps=488282

python train.py --gpu_num=2 --actv_name="blocks.7.attn.hook_k" --beta2=0.9999 --layer=7 --head=3 --l1_coefficient=0.005 --allow_lower_decoder_norm=True --batch_size=512 --d_hidden=1000 --start_lr_decay=440000 --train_steps=488282
python train.py --gpu_num=2 --actv_name="blocks.7.attn.hook_k" --beta2=0.9999 --layer=7 --head=9 --l1_coefficient=0.005 --allow_lower_decoder_norm=True --batch_size=512 --d_hidden=1000 --start_lr_decay=440000 --train_steps=488282
python train.py --gpu_num=2 --actv_name="blocks.8.attn.hook_k" --beta2=0.9999 --layer=8 --head=6 --l1_coefficient=0.005 --allow_lower_decoder_norm=True --batch_size=512 --d_hidden=1000 --start_lr_decay=440000 --train_steps=488282
python train.py --gpu_num=2 --actv_name="blocks.8.attn.hook_k" --beta2=0.9999 --layer=8 --head=10 --l1_coefficient=0.005 --allow_lower_decoder_norm=True --batch_size=512 --d_hidden=1000 --start_lr_decay=440000 --train_steps=488282


# python train.py --gpu_num=2 --actv_name="blocks.9.attn.hook_z" --layer=9 --head=6 --l1_coefficient=0.006
# python train.py --gpu_num=2 --actv_name="blocks.9.attn.hook_z" --layer=9 --head=7 --l1_coefficient=0.006
# python train.py --gpu_num=2 --actv_name="blocks.9.attn.hook_z" --layer=9 --head=8 --l1_coefficient=0.006
# python train.py --gpu_num=2 --actv_name="blocks.9.attn.hook_z" --layer=9 --head=9 --l1_coefficient=0.006
# python train.py --gpu_num=2 --actv_name="blocks.9.attn.hook_z" --layer=9 --head=10 --l1_coefficient=0.006
# python train.py --gpu_num=2 --actv_name="blocks.9.attn.hook_z" --layer=9 --head=11 --l1_coefficient=0.006