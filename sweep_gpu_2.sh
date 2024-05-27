# name mover values
python ./research/sweep.py --target_metric=0.8 --output_file='search_results.csv' --component_name="v" --layer=9 --head=9 --gpu_num=2
python ./research/sweep.py --target_metric=0.8 --output_file='search_results.csv' --component_name="v" --layer=9 --head=6 --gpu_num=2
python ./research/sweep.py --target_metric=0.8 --output_file='search_results.csv' --component_name="v" --layer=10 --head=0 --gpu_num=2
# name mover keys
python ./research/sweep.py --target_metric=0.8 --output_file='search_results.csv' --component_name="k" --layer=9 --head=9 --gpu_num=2
python ./research/sweep.py --target_metric=0.8 --output_file='search_results.csv' --component_name="k" --layer=9 --head=6 --gpu_num=2
python ./research/sweep.py --target_metric=0.8 --output_file='search_results.csv' --component_name="k" --layer=10 --head=0 --gpu_num=2
# S-inhibition heads values
python ./research/sweep.py --target_metric=0.8 --output_file='search_results.csv' --component_name="v" --layer=7 --head=3 --gpu_num=2
python ./research/sweep.py --target_metric=0.8 --output_file='search_results.csv' --component_name="v" --layer=7 --head=9 --gpu_num=2