#python main.py --gpu_num=3 --actv_name="blocks.9.attn.hook_z" --layer=9 --head=9 --l1_coefficient=0.025 --allow_lower_decoder_norm=True --batch_size=512 --d_hidden=1000 --start_lr_decay=440000 --train_steps=488282 --use_disc_dataset=True
#python main.py --gpu_num=3 --actv_name="blocks.9.attn.hook_z" --layer=9 --head=6 --l1_coefficient=0.025 --allow_lower_decoder_norm=True --batch_size=512 --d_hidden=1000 --start_lr_decay=440000 --train_steps=488282
#python main.py --gpu_num=3 --actv_name="blocks.10.attn.hook_z" --layer=10 --head=0 --l1_coefficient=0.025 --allow_lower_decoder_norm=True --batch_size=512 --d_hidden=1000  --start_lr_decay=440000 --train_steps=488282

python train.py --gpu_num=3 --actv_name="blocks.9.attn.hook_k" --layer=9 --head=9 --l1_coefficient=0.01 --allow_lower_decoder_norm=True --batch_size=512 --d_hidden=1000 --start_lr_decay=440000 --train_steps=488282
python train.py --gpu_num=3 --actv_name="blocks.9.attn.hook_k" --layer=9 --head=6 --l1_coefficient=0.01 --allow_lower_decoder_norm=True --batch_size=512 --d_hidden=1000 --start_lr_decay=440000 --train_steps=488282
python train.py --gpu_num=3 --actv_name="blocks.10.attn.hook_k" --layer=10 --head=0 --l1_coefficient=0.01 --allow_lower_decoder_norm=True --batch_size=512 --d_hidden=1000 --start_lr_decay=440000 --train_steps=488282

python train.py --gpu_num=3 --actv_name="blocks.9.attn.hook_v" --layer=9 --head=9 --l1_coefficient=0.01 --allow_lower_decoder_norm=True --batch_size=512 --d_hidden=1000 --start_lr_decay=440000 --train_steps=488282
python train.py --gpu_num=3 --actv_name="blocks.9.attn.hook_v" --layer=9 --head=6 --l1_coefficient=0.01 --allow_lower_decoder_norm=True --batch_size=512 --d_hidden=1000 --start_lr_decay=440000 --train_steps=488282
python train.py --gpu_num=3 --actv_name="blocks.10.attn.hook_v" --layer=10 --head=0 --l1_coefficient=0.01 --allow_lower_decoder_norm=True --batch_size=512 --d_hidden=1000 --start_lr_decay=440000 --train_steps=488282


#python main.py --gpu_num=3 --actv_name="blocks.9.attn.hook_z" --layer=9 --head=0 --l1_coefficient=0.006

#python main.py --gpu_num=3 --actv_name="blocks.9.attn.hook_z" --layer=9 --head=1 --l1_coefficient=0.006

#python main.py --gpu_num=3 --actv_name="blocks.9.attn.hook_z" --layer=9 --head=2 --l1_coefficient=0.006

#python main.py --gpu_num=3 --actv_name="blocks.9.attn.hook_z" --layer=9 --head=3 --l1_coefficient=0.006

#python main.py --gpu_num=3 --actv_name="blocks.9.attn.hook_z" --layer=9 --head=4 --l1_coefficient=0.006

#python main.py --gpu_num=3 --actv_name="blocks.9.attn.hook_z" --layer=9 --head=5 --l1_coefficient=0.006


