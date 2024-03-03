# python train.py --gpu_num=2 --actv_name="blocks.10.attn.hook_q" --layer=10 --head=0 --l1_coefficient=0.006
# python train.py --gpu_num=2 --actv_name="blocks.10.attn.hook_q" --layer=10 --head=1 --l1_coefficient=0.006
# python train.py --gpu_num=2 --actv_name="blocks.10.attn.hook_q" --layer=10 --head=2 --l1_coefficient=0.006
# python train.py --gpu_num=2 --actv_name="blocks.10.attn.hook_q" --layer=10 --head=10 --l1_coefficient=0.006
# python train.py --gpu_num=2 --actv_name="blocks.11.attn.hook_q" --layer=11 --head=2 --l1_coefficient=0.006
# python train.py --gpu_num=2 --actv_name="blocks.11.attn.hook_q" --layer=11 --head=9 --l1_coefficient=0.006
# python train.py --gpu_num=2 --actv_name="blocks.9.attn.hook_q" --layer=9 --head=7 --l1_coefficient=0.006

# values

python train.py --gpu_num=0 --actv_name="blocks.9.attn.hook_q" --layer=9 --head=9 --l1_coefficient=0.01 --allow_lower_decoder_norm=True --batch_size=512 --d_hidden=1000 --start_lr_decay=440000 --train_steps=488282
python train.py --gpu_num=0 --actv_name="blocks.9.attn.hook_q" --layer=9 --head=6 --l1_coefficient=0.01 --allow_lower_decoder_norm=True --batch_size=512 --d_hidden=1000 --start_lr_decay=440000 --train_steps=488282
python train.py --gpu_num=0 --actv_name="blocks.10.attn.hook_q" --layer=10 --head=0 --l1_coefficient=0.01 --allow_lower_decoder_norm=True --batch_size=512 --d_hidden=1000 --start_lr_decay=440000 --train_steps=488282




# python train.py --gpu_num=0 --actv_name="blocks.9.attn.hook_z" --layer=9 --head=9 --l1_coefficient=0.006 --use_disc_dataset True --allow_lower_decoder_norm False
#
# python train.py --gpu_num=2 --actv_name="blocks.9.attn.hook_z" --layer=9 --head=9 --l1_coefficient=0.009 --use_disc_dataset=True --allow_lower_decoder_norm=True
#
# python train.py --gpu_num=3 --actv_name="blocks.9.attn.hook_z" --layer=9 --head=9 --l1_coefficient=0.009 --use_disc_dataset=True --allow_lower_decoder_norm=True