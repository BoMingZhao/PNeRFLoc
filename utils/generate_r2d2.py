import os
import pandas as pd
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate parameters")
    parser.add_argument('--path', type=str, default="./data_src/")
    parser.add_argument('--save_path', type=str, default="./descriptor/r2d2/imgs")
    parser.add_argument('--dataset', type=str, default="Replica", choices=["Replica", "7Scenes"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument('--skip', type=int, default=10)
    args = parser.parse_args()

    if args.dataset == "Replica":
        scenes = {"room0": 2000, "room1": 2000, "room2": 2000, "office0": 2000, "office1": 2000, "office2": 2000, "office3": 2000, "office4": 2000}
    elif args.dataset == "7Scenes":
        scenes = {"chess": 4000, "pumpkin": 4000, "office": 6000, "stairs": 2000, "heads": 1000, "fire": 2000, "redkitchen": 7000}
    path = os.path.join(args.path, args.dataset)
    save_path = os.path.join(args.save_path, args.dataset)
    os.makedirs(save_path, exist_ok=True)
    for scene in scenes.keys():
        train_id_list = []
        skip = args.skip if scenes[scene] > 1000 else 1
        for i in range(0, scenes[scene], skip):
            img_path = os.path.join(path, scene, "exported", "color", f'{i:d}.jpg')
            train_id_list.append([os.path.abspath(img_path)])
        output_data = pd.DataFrame(train_id_list)
        output_data.to_csv(os.path.join(save_path, scene + ".txt"), sep=' ', index=False, header=False)

        test_id_list = []
        length = len(os.listdir(os.path.join(path, scene, "exported", "pose"))) 
        for i in range(scenes[scene], length):
            img_path = os.path.join(path, scene, "exported", "color", f'{i:d}.jpg')
            test_id_list.append([os.path.abspath(img_path)])
        output_data = pd.DataFrame(test_id_list)
        output_data.to_csv(os.path.join(save_path, scene + "_query.txt"), sep=' ', index=False, header=False)

    os.chdir("descriptor/r2d2")
    for scene in scenes.keys():
        os.system(f"python extract.py --model models/r2d2_WASF_N8_big.pt --images imgs/{args.dataset}/{scene}_query.txt --scale-f 5 --max-size 2000 --top-k 800 --gpu {args.gpu}")
    
