echo "Generate Replica key points"
python utils/generate_r2d2.py --dataset "Replica" --gpu 0

wait

echo "Generate 7Scenes key points"
python utils/generate_r2d2.py --dataset "7Scenes" --gpu 0
wait 

echo "finish"