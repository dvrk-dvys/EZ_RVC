import torch

from vencoder.encoder import SpeechEncoder
from vencoder.hubert import hubert_model
import os
import importlib


class HubertSoft(SpeechEncoder):
    def __init__(self, vec_path="./model_dir/pretrain/hubert-soft-0d54a1f4.pt", device=None):
        super().__init__()
        print("load model(s) from {}".format(vec_path))

        def find_project_root(target_file):
            """Find the root directory of the project based on the existence of a specific target file."""
            current_path = os.path.dirname(os.path.abspath(__file__))  # Start from the current script location
            while not os.path.isfile(os.path.join(current_path, target_file)):
                current_path = os.path.dirname(current_path)
                if current_path == "/":
                    raise FileNotFoundError(f"Cannot find the root directory based on the target file: {target_file}")
            return current_path

        def is_running_in_colab():
            if importlib.util.find_spec('google.colab') is not None:
                return True
            else:
                return False

        if is_running_in_colab == True:
            vec_path = "/content/EZ_RVC/model_dir/pretrain/hubert-soft-0d54a1f4.pt"
        # # The directory where the script should run
        # desired_directory = "/EZ_RVC"
        # Get the current working directory
        current_directory = os.getcwd()

        # if current_directory != desired_directory:
        #     os.chdir(desired_directory)

        # print(f"Script is now running in: {os.getcwd()}")
        #
        # root_directory = find_project_root('inference_main.py')
        # # Change to the root directory
        # os.chdir(root_directory)

        # # Ensure the directory exists
        # model_directory = "/content/EZ_RVC/logs/44k/"
        # os.makedirs(model_directory, exist_ok=True)


        hubert_soft = hubert_model.hubert_soft(vec_path)
        if device is None:
            # self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.dev = torch.device("mps" if torch.cuda.is_available() else "cpu")
        else:
            self.dev = torch.device(device)
        self.hidden_dim = 256
        self.model = hubert_soft.to(self.dev)

    def encoder(self, wav):
        feats = wav
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats[None,None,:]  
        with torch.no_grad():
            with torch.inference_mode():
                units = self.model.units(feats)
                return units.transpose(1,2)
