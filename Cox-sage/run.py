import subprocess
import os
import argparse
import sys
import pickle as pkl

def remove_dir(path):
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(path)

if __name__ == '__main__':
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Training script parameters.')

    # Adding arguments with no default for interpreter_path
    parser.add_argument('--interpreter_path', type=str,
                        help='Path to the Python interpreter (leave empty to use current interpreter).')
    parser.add_argument('--data_path', type=str, default='dataset/data',
                        help='Path to the data directory.')
    parser.add_argument('--save_path', type=str, default='save',
                        help='Path to the directory where models will be saved.')
    parser.add_argument('--log_path', type=str, default='log.txt',
                        help='Path to the log file.')
    parser.add_argument('--disease', type=str, default='LIHC',
                        help='Disease Type.')
    parser.add_argument('--seed', type=str, default='1',
                        help='Disease Type.')
    # Parse the arguments
    args = parser.parse_args()

    # Use the current interpreter path if not provided
    interpreter_path = args.interpreter_path if args.interpreter_path else sys.executable
    data_path = args.data_path
    save_path = args.save_path
    log_path = args.log_path
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # Recording train log
    with open(log_path, 'a') as f:
        f.write('')

    cwd = os.getcwd()
    print(f'Current working directory: {cwd}')

    diseases = [args.disease]
    layers = [1, 2, 4]
    seed_ns = [0,1,2,3,4]
    list_save_root = "prediction_save"
    if not os.path.exists("prediction_save"):
        os.mkdir(list_save_root)
    # Outer loop processes each disease in order
    # Dowloading dataset
    subprocess.run([interpreter_path,"data_download.py"])

    diseases = os.listdir("dataset/data")
    for disease in diseases:
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        print(f'Training prognostic model for {disease}...')
        # Inner loop executes different layers in parallel
        for l in layers:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            print(f'layers: {l}...')
            list_save_path = f"{list_save_root}/{disease}_{l}layers.pkl"
            with open(list_save_path,'wb') as f:
                obj = []
                pkl.dump(obj,f)
            for seed_n in seed_ns:
                prediction_path = f"{list_save_root}/{disease}_{l}layers_{seed_n}seed"
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                print(f'seed_n:{seed_n}')
                if disease not in ['LIHC','ESCA']:
                    subprocess.run([
                        interpreter_path, "train_single_model.py",
                        "--data_root", data_path,
                        "--model_save_root", save_path,
                        "--disease", disease,
                        "--num_layers", str(l),
                        "--log_path", log_path,
                        "--seed_n",str(seed_n),
                        "--list_path",list_save_path,
                        "--file_path", prediction_path
                        ])
                elif disease=='LIHC':
                    subprocess.run([
                        interpreter_path, "train_single_model.py",
                        "--data_root", data_path,
                        "--model_save_root", save_path,
                        "--disease", disease,
                        "--num_layers", str(l),
                        "--log_path", log_path,
                        "--seed",str(17),
                        "--seed_n",str(seed_n),
                        "--list_path",list_save_path,
                        "--file_path", prediction_path
                    ])
                else:
                    subprocess.run([
                        interpreter_path, "train_single_model.py",
                        "--data_root", data_path,
                        "--model_save_root", save_path,
                        "--disease", disease,
                        "--num_layers", str(l),
                        "--log_path", log_path,
                        "--seed", str(2),
                        "--seed_n",str(seed_n),
                        "--list_path",list_save_path,
                        "--file_path", prediction_path
                    ])
                remove_dir(save_path)







