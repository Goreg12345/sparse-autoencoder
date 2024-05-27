# name mover outputs
python ./research/sweep.py --target_metric=0.8 --output_file='search_results.csv' --component_name="z" --layer=9 --head=9 --gpu_num=3
python ./research/sweep.py --target_metric=0.8 --output_file='search_results.csv' --component_name="z" --layer=9 --head=6 --gpu_num=3
python ./research/sweep.py --target_metric=0.8 --output_file='search_results.csv' --component_name="z" --layer=10 --head=0 --gpu_num=3
# name mover queries
python ./research/sweep.py --target_metric=0.8 --output_file='search_results.csv' --component_name="q" --layer=9 --head=9 --gpu_num=3
python ./research/sweep.py --target_metric=0.8 --output_file='search_results.csv' --component_name="q" --layer=9 --head=6 --gpu_num=3
python ./research/sweep.py --target_metric=0.8 --output_file='search_results.csv' --component_name="q" --layer=10 --head=0 --gpu_num=3
# si values
python ./research/sweep.py --target_metric=0.8 --output_file='search_results.csv' --component_name="v" --layer=8 --head=6 --gpu_num=3
python ./research/sweep.py --target_metric=0.8 --output_file='search_results.csv' --component_name="v" --layer=8 --head=10 --gpu_num=3