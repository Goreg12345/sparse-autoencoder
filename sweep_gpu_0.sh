# duplicate token heads
python ./research/sweep.py --target_metric=0.8 --output_file='search_results.csv' --component_name="z" --layer=0 --head=1 --gpu_num=0
python ./research/sweep.py --target_metric=0.8 --output_file='search_results.csv' --component_name="z" --layer=3 --head=0 --gpu_num=0
python ./research/sweep.py --target_metric=0.8 --output_file='search_results.csv' --component_name="z" --layer=0 --head=10 --gpu_num=0
# induction heads
python ./research/sweep.py --target_metric=0.8 --output_file='search_results.csv' --component_name="z" --layer=5 --head=5 --gpu_num=0
python ./research/sweep.py --target_metric=0.8 --output_file='search_results.csv' --component_name="z" --layer=6 --head=9 --gpu_num=0
python ./research/sweep.py --target_metric=0.8 --output_file='search_results.csv' --component_name="z" --layer=5 --head=8 --gpu_num=0
python ./research/sweep.py --target_metric=0.8 --output_file='search_results.csv' --component_name="z" --layer=5 --head=9 --gpu_num=0
# S-inhibition heads out
python ./research/sweep.py --target_metric=0.8 --output_file='search_results.csv' --component_name="z" --layer=7 --head=3 --gpu_num=0
python ./research/sweep.py --target_metric=0.8 --output_file='search_results.csv' --component_name="z" --layer=7 --head=9 --gpu_num=0
python ./research/sweep.py --target_metric=0.8 --output_file='search_results.csv' --component_name="z" --layer=8 --head=6 --gpu_num=0
python ./research/sweep.py --target_metric=0.8 --output_file='search_results.csv' --component_name="z" --layer=8 --head=10 --gpu_num=0