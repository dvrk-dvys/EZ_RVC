#%%
#@title Clone repository and install requirements

#@markdown # Clone repository and install requirements

#@markdown

#@markdown ### After the execution is completed, the runtime will **automatically restart**

#@markdown

!git clone https://github.com/dvrk-dvys/EZ_RVC.git
%cd /content/EZ_RVC
%pip install --upgrade pip setuptools
%pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
exit()
#%%
#@title Mount google drive and select which directories to sync with google drive
#@markdown # Mount google drive and select which directories to sync with google drive
#@markdown

from google.colab import drive
drive.mount("/content/drive", force_remount=True)

#@markdown Directory to store **necessary files**, dont miss the slash at the end👇.
EZ_RVC_data_dir = "/content/drive/MyDrive/dataset"  #@param {type:"string"}
#@markdown By default it will create a `sovits4data/` folder in your google drive.
RAW_DIR = EZ_RVC_data_dir + "raw/"
RESULTS_DIR = EZ_RVC_data_dir + "results/"
LOGS_DIR = EZ_RVC_data_dir + "logs/"
SR_DIR = EZ_RVC_data_dir + "logs/44k/"


#@markdown ### These folders will be synced with your google drvie
#@markdown　### **Strongly recommend to check all.**
#@markdown Sync **input audios** and **output audios**
# sync_raw_and_results = True  #@param {type:"boolean"}
# if sync_raw_and_results:
#   # !mkdir -p {RAW_DIR}
#   # !mkdir -p {RESULTS_DIR}
#   !rm -rf /content/EZ_RVC/raw
#   !rm -rf /content/EZ_RVC/results
#   !ln -s {RAW_DIR} /content/EZ_RVC/raw
#   !ln -s {RESULTS_DIR} /content/EZ_RVC/results

#@markdown Sync **config** and **models**
# sync_configs_and_logs = True  #@param {type:"boolean"}
# if sync_configs_and_logs:
#     !mkdir -p {LOGS_DIR}



#%%
#@title Get pretrained model(Optional but strongly recommend).

#@markdown # Get pretrained model(Optional but strongly recommend).

#@markdown

#@markdown - Pre-trained model files: `G_0.pth` `D_0.pth`
#@markdown   - Place them under /sovits4data/logs/44k/ in your google drive manualy

#@markdown Get them from svc-develop-team(TBD) or anywhere else.

#@markdown Although the pretrained model generally does not cause any copyright problems, please pay attention to it. For example, ask the author in advance, or the author has indicated the feasible use in the description clearly.

download_pretrained_model = True #@param {type:"boolean"}
G_0_URL = "https://drive.google.com/file/d/1-3LqFaS9E-s40tg6yxkZgjlb9mxsJm67/view?usp=drive_link" # @param ["https://huggingface.co/datasets/ms903/sovits4.0-768vec-layer12/resolve/main/sovits_768l12_pre_large_320k/clean_G_320000.pth", "https://huggingface.co/1asbgdh/sovits4.0-volemb-vec768/resolve/main/clean_G_320000.pth", "https://huggingface.co/datasets/ms903/sovits4.0-768vec-layer12/resolve/main/vol_emb/clean_G_320000.pth", "https://drive.google.com/file/d/1-iRmVbvURPsVNS8KPwN2BjAouwAVj0af/view?usp=drive_link"] {allow-input: true}
D_0_URL = "https://drive.google.com/file/d/1-5At1wywypCMDY5s3XxLj8uSC9sBwIks/view?usp=drive_link" # @param ["https://huggingface.co/datasets/ms903/sovits4.0-768vec-layer12/resolve/main/sovits_768l12_pre_large_320k/clean_D_320000.pth", "https://huggingface.co/1asbgdh/sovits4.0-volemb-vec768/resolve/main/clean_D_320000.pth", "https://huggingface.co/datasets/ms903/sovits4.0-768vec-layer12/resolve/main/vol_emb/clean_D_320000.pth", "https://drive.google.com/file/d/1-rtgMwfsWyBVuZVpoIxtTrdLHkEmtm_m/view?usp=drive_link"] {allow-input: true}

download_pretrained_diffusion_model = False #@param {type:"boolean"}
diff_model_URL = "https://huggingface.co/datasets/ms903/Diff-SVC-refactor-pre-trained-model/resolve/main/fix_pitch_add_vctk_600k/model_0.pt" #@param {type:"string"}

%cd /content/EZ_RVC

if download_pretrained_model:
    !curl -L {D_0_URL} -o logs/44k/D_0.pth
    !md5sum logs/44k/D_0.pth
    !curl -L {G_0_URL} -o logs/44k/G_0.pth
    !md5sum logs/44k/G_0.pth

if download_pretrained_diffusion_model:
    !mkdir -p logs/44k/diffusion
    !curl -L {diff_model_URL} -o logs/44k/diffusion/model_0.pt
    !md5sum logs/44k/diffusion/model_0.pt
#%%
!ls
#%%
#@title Start training

#@markdown # Start training

#@markdown If you want to use pre-trained models, upload them to /sovits4data/logs/44k/ in your google drive manualy.

#@markdown

# %cd /content/EZ_RVC
# %cd /content/
!pip install faiss-cpu


#@markdown Whether to enable tensorboard
tensorboard_on = True  #@param {type:"boolean"}

if tensorboard_on:
  %load_ext tensorboard
  %tensorboard --logdir logs/44k


config_path = "/content/EZ_RVC/dataset/configs/config_colab.json"
# !ln -s "/content/drive/MyDrive/dataset" "/content/EZ_RVC/dataset"

# from model_dir.pretrain.meta import get_speech_encoder
# url, output = get_speech_encoder(config_path)

# import os
# if not os.path.exists(output):
#   !curl -L {url} -o {output}

!python train.py -c {config_path} -m 44k -md "/content/drive/MyDrive/dataset/44k/44k/sza"
#%%
!ls
#%% md
# # **Inference**
# ### Upload wav files from this notebook
# ### **OR**
# ### Upload to `sovits4data/raw/` in your google drive manualy (should be faster)
#%%
%cd /
# !ls content/drive/MyDrive/dataset/44k/raw/PinkPantheress_Ice_Spice_Boys_a_liar_Almost_Studio_Acapella.wav
# !ls content/drive/MyDrive/dataset/44k/raw/Unforgettable_Nat_King_Cole_isolated_vocal.wav
!ls content/drive/MyDrive/dataset/44k/raw/Aaliyah_4_Page_Letter_A_Capella.wav

%cd /content/EZ_RVC
# %cd content/EZ_RVC
# !ls content/drive/MyDrive/dataset/44k/raw/PinkPantheress_Ice_Spice_Boys_a_liar_Almost_Studio_Acapella.wav

# !ls model_dir/pretrain/nsf_hifigan/nsf_hifigan/model\
# !ls drive/MyDrive/dataset/44k/44k/D_50000.pth

# with opx
#%%
#@title Start inference (and download)

#@markdown # Start inference (and download)

#@markdown Parameters see [README.MD#Inference](https://github.com/svc-develop-team/so-vits-svc#-inference)

#@markdown

wav_filename = "Aaliyah_4_Page_Letter_A_Capella"  #@param {type:"string"}
# model_filename = "G_50000"  #@param {type:"string"}
model_path = "/content/drive/MyDrive/dataset/44k/44k/sza/G_10000.pth" #@param {type:"string"}
speaker = "sza_singing"  #@param {type:"string"}
trans = "0"  #@param {type:"string"}
cluster_infer_ratio = "0"  #@param {type:"string"}
auto_predict_f0 = True  #@param {type:"boolean"}
apf = ""
if auto_predict_f0:
  apf = " -a "

f0_predictor = "crepe" #@param ["crepe", "pm", "dio", "harvest", "rmvpe", "fcpe"]

enhance = True  #@param {type:"boolean"}
ehc = ""
if enhance:
  ehc = " -eh "
#@markdown

#@markdown Generally keep default:
# config_filename = "config_colab.json"  #@param {type:"string"}
# config_path = "/content/EZ_RVC/configs/" + config_filename
config_path = "/content/EZ_RVC/dataset/configs/config_colab.json"

from app.models.pretrain import get_speech_encoder
url, output = get_speech_encoder(config_path)

import os

if f0_predictor == "rmvpe" and not os.path.exists("./pretrain/rmvpe.pt"):
  !curl -L https://huggingface.co/datasets/ylzz1997/rmvpe_pretrain_model/resolve/main/rmvpe.pt -o pretrain/rmvpe.pt

if f0_predictor == "fcpe" and not os.path.exists("./pretrain/fcpe.pt"):
  !curl -L https://huggingface.co/datasets/ylzz1997/rmvpe_pretrain_model/resolve/main/fcpe.pt -o pretrain/fcpe.pt

if not os.path.exists(output):
  !curl -L {url} -o {output}

kmeans_filenname = "kmeans_10000.pt"  #@param {type:"string"}
kmeans_path = "/content/EZ_RVC/logs/44k/" + kmeans_filenname
slice_db = "-40"  #@param {type:"string"}
wav_format = "flac"  #@param {type:"string"}

key = "auto" if auto_predict_f0 else f"{trans}key"
cluster_name = "" if cluster_infer_ratio == "0" else f"_{cluster_infer_ratio}"
isdiffusion = "sovits"
wav_output = f"/content/EZ_RVC/results/{wav_filename}_{key}_{speaker}{cluster_name}_{isdiffusion}_{f0_predictor}.{wav_format}"

%cd /content/EZ_RVC/
!pip install faiss-cpu
!python inference_main.py -n {wav_filename} -m {model_path} -s {speaker} -t {trans} -cr {cluster_infer_ratio} -c {config_path} -cm {kmeans_path} -sd {slice_db} -wf {wav_format} {apf} --f0_predictor={f0_predictor} {ehc}

#@markdown

#@markdown If you dont want to download from here, uncheck this.
download_after_inference = True  #@param {type:"boolean"}

if download_after_inference:
  from google.colab import files
  files.download(wav_output)