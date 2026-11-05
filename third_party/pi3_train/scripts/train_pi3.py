import os
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sys
sys.path.append('.')

import hydra
import trainers

@hydra.main(version_base="1.2", config_path="../configs", config_name="peft_3radapter_highres")
def main(hydra_cfg):
    trainer = eval(hydra_cfg.trainer)(hydra_cfg)
    trainer.train()

if __name__ == '__main__':
    main()