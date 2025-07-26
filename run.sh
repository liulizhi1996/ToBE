## Step 1: Preprocess
cd src/preprocess
for dataset in Yelp Gowalla AmazonBook Tmall Movielens1M AlibabaIFashion; do
    python3 preprocess.py --dataset $dataset --do_filtering
done

## Step 2: Recommendation
cd src/task
python3 recommend.py --dataset Yelp --lr 0.0005 --lambda_reg 0.1 --lambda_au 0.5 --train_batch_size 2048 --num_layers 3
python3 recommend.py --dataset Gowalla --lr 0.001 --lambda_reg 0.1 --lambda_au 0.5 --train_batch_size 2048 --num_layers 3
python3 recommend.py --dataset AmazonBook --lr 0.001 --lambda_reg 0.1 --lambda_au 1 --train_batch_size 1024 --num_layers 3
python3 recommend.py --dataset Tmall --lr 0.001 --lambda_reg 0.1 --lambda_au 10 --train_batch_size 2048 --num_layers 3
python3 recommend.py --dataset Movielens1M --lr 0.0005 --lambda_reg 0.1 --lambda_au 0.5 --train_batch_size 2048 --num_layers 3
python3 recommend.py --dataset AlibabaIFashion --lr 0.001 --lambda_reg 0.2 --lambda_au 0.5 --train_batch_size 2048 --num_layers 3
