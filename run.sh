seed=1
epochs=1000
hidden_num=256
# the influence of different hidden_num

# python main.py --method mlp  --seed $seed --lr 0.1   --epochs $epochs --hidden_number $hidden_num
# python main.py --method mlp  --seed $seed --lr 0.01   --epochs $epochs --hidden_number $hidden_num
# python main.py --method mlp  --seed $seed --lr 0.001   --epochs $epochs --hidden_number $hidden_num
python main.py --method mlqp2  --seed $seed --lr 0.01   --epochs $epochs --hidden_number $hidden_num