bash ./dev_scripts/train/Replica/room1.sh &
bash ./dev_scripts/train/Replica/room2.sh

wait

bash ./dev_scripts/train/Replica/office0.sh &
bash ./dev_scripts/train/Replica/office1.sh

wait

bash ./dev_scripts/train/Replica/office2.sh &
bash ./dev_scripts/train/Replica/office3.sh

wait

bash ./dev_scripts/train/Replica/office4.sh &
bash ./dev_scripts/train/Replica/room0.sh

wait

echo "Finish"