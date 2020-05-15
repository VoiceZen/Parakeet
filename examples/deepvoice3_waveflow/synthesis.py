# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import division
import os
import argparse
import ruamel.yaml
import numpy as np
import soundfile as sf

from paddle import fluid
import paddle.fluid.layers as F
import paddle.fluid.dygraph as dg
from tensorboardX import SummaryWriter

from parakeet.g2p import en
from parakeet.modules.weight_norm import WeightNormWrapper
from parakeet.utils.layer_tools import summary
from parakeet.utils import io

from utils import make_model, eval_model, plot_alignment

import time
import random
import librosa

from parakeet.models.waveflow import waveflow_modules
from parakeet.modules import weight_norm

def add_config_options_to_parser(parser):
    parser.add_argument(
        '--valid_size', type=int, help="size of the valid dataset")
    parser.add_argument(
        '--segment_length',
        type=int,
        help="the length of audio clip for training")
    parser.add_argument(
        '--sample_rate', type=int, help="sampling rate of audio data file")
    parser.add_argument(
        '--fft_window_shift',
        type=int,
        help="the shift of fft window for each frame")
    parser.add_argument(
        '--fft_window_size',
        type=int,
        help="the size of fft window for each frame")
    parser.add_argument(
        '--fft_size', type=int, help="the size of fft filter on each frame")
    parser.add_argument(
        '--mel_bands',
        type=int,
        help="the number of mel bands when calculating mel spectrograms")
    parser.add_argument(
        '--mel_fmin',
        type=float,
        help="lowest frequency in calculating mel spectrograms")
    parser.add_argument(
        '--mel_fmax',
        type=float,
        help="highest frequency in calculating mel spectrograms")

    parser.add_argument(
        '--seed', type=int, help="seed of random initialization for the model")
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument(
        '--batch_size', type=int, help="batch size for training")
    parser.add_argument(
        '--test_every', type=int, help="test interval during training")
    parser.add_argument(
        '--save_every',
        type=int,
        help="checkpointing interval during training")
    parser.add_argument(
        '--max_iterations', type=int, help="maximum training iterations")

    parser.add_argument(
        '--sigma',
        type=float,
        help="standard deviation of the latent Gaussian variable")
    parser.add_argument('--n_flows', type=int, help="number of flows")
    parser.add_argument(
        '--n_group',
        type=int,
        help="number of adjacent audio samples to squeeze into one column")
    parser.add_argument(
        '--n_layers',
        type=int,
        help="number of conv2d layer in one wavenet-like flow architecture")
    parser.add_argument(
        '--n_channels', type=int, help="number of residual channels in flow")
    parser.add_argument(
        '--kernel_h',
        type=int,
        help="height of the kernel in the conv2d layer")
    parser.add_argument(
        '--kernel_w', type=int, help="width of the kernel in the conv2d layer")

    parser.add_argument('--config', type=str, help="Path to the config file.")



class Config():
  def __init__(self, **entries):
    self.__dict__.update(entries)
    
    
class WaveFlow():
  def __init__(self,
               config,checkpoint_dir,
               parallel=False,
               rank=0,
               nranks=1,
               tb_logger=None):
    self.config = config
    self.checkpoint_dir = checkpoint_dir
    self.parallel = parallel
    self.rank = rank
    self.nranks = nranks
    self.tb_logger = tb_logger
    # self.dtype = "float16" if config.use_fp16 else "float32"
    self.dtype = "float32"

  def build(self):
    """Initialize the model.

    Args:
        training (bool, optional): Whether the model is built for training or inference.
            Defaults to True.

    Returns:
        None
    """
    config = self.config
    # dataset = LJSpeech(config, self.nranks, self.rank)
    # self.trainloader = dataset.trainloader
    # self.validloader = dataset.validloader

    waveflow = waveflow_modules.WaveFlowModule(config)

    
    config.iteration=None
    iteration = io.load_parameters(
                model=waveflow,
                checkpoint_dir=self.checkpoint_dir,
                iteration=self.config.iteration,
                checkpoint_path=self.config.checkpoint)
    print("Rank {}: checkpoint loaded.".format(self.rank))

    for layer in waveflow.sublayers():
      if isinstance(layer, weight_norm.WeightNormWrapper):
        layer.remove_weight_norm()    
    self.waveflow = waveflow
    
    return iteration

  @dg.no_grad
  def infer(self, mel):
    self.waveflow.eval()
    config = self.config
    print(mel.shape, 'mel.shape')
    start_time = time.time()
    audio = self.waveflow.synthesize(mel, sigma=self.config.sigma)
    syn_time = time.time() - start_time

    
    return audio,start_time,syn_time



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Synthsize waveform with a checkpoint.")
    parser.add_argument("--config", type=str, help="experiment config")
    parser.add_argument("--device", type=int, default=-1, help="device to use")

    g = parser.add_mutually_exclusive_group()
    g.add_argument("--checkpoint", type=str, help="checkpoint to resume from")
    g.add_argument(
        "--iteration",
        type=int,
        help="the iteration of the checkpoint to load from output directory")
    
    parser.add_argument("--waveflow_config", type=str, help="waveflow config")
    # add_config_options_to_parser(parser)

    wv = parser.add_mutually_exclusive_group()
    wv.add_argument("--waveflow_checkpoint_dir", type=str, help="checkpoint to resume from",
                  default="")
                  #default="~/users/ak47/projects/Parakeet/pretrained/waveflow_res128_ljspeech_ckpt_1.0"
    wv.add_argument(
        "--waveflow_iteration",
        type=int,
        help="the iteration of the checkpoint to load from output directory")
    

    parser.add_argument("text", type=str, help="text file to synthesize")
    parser.add_argument(
        "output", type=str, help="path to save synthesized audio")
    
    args = parser.parse_args()
    with open(args.config, 'rt') as f:
        config = ruamel.yaml.safe_load(f)

    print("Command Line Args: ")
    for k, v in vars(args).items():
        print("{}: {}".format(k, v))

    if args.device == -1:
        place = fluid.CPUPlace()
    else:
        place = fluid.CUDAPlace(args.device)

    with dg.guard(place):
        # =========================model=========================
        transform_config = config["transform"]
        replace_pronounciation_prob = transform_config[
            "replace_pronunciation_prob"]
        sample_rate = transform_config["sample_rate"]
        preemphasis = transform_config["preemphasis"]
        n_fft = transform_config["n_fft"]
        n_mels = transform_config["n_mels"]

        model_config = config["model"]
        downsample_factor = model_config["downsample_factor"]
        r = model_config["outputs_per_step"]
        n_speakers = model_config["n_speakers"]
        speaker_dim = model_config["speaker_embed_dim"]
        speaker_embed_std = model_config["speaker_embedding_weight_std"]
        n_vocab = en.n_vocab
        embed_dim = model_config["text_embed_dim"]
        linear_dim = 1 + n_fft // 2
        use_decoder_states = model_config[
            "use_decoder_state_for_postnet_input"]
        filter_size = model_config["kernel_size"]
        encoder_channels = model_config["encoder_channels"]
        decoder_channels = model_config["decoder_channels"]
        converter_channels = model_config["converter_channels"]
        dropout = model_config["dropout"]
        padding_idx = model_config["padding_idx"]
        embedding_std = model_config["embedding_weight_std"]
        max_positions = model_config["max_positions"]
        freeze_embedding = model_config["freeze_embedding"]
        trainable_positional_encodings = model_config[
            "trainable_positional_encodings"]
        use_memory_mask = model_config["use_memory_mask"]
        query_position_rate = model_config["query_position_rate"]
        key_position_rate = model_config["key_position_rate"]
        window_backward = model_config["window_backward"]
        window_ahead = model_config["window_ahead"]
        key_projection = model_config["key_projection"]
        value_projection = model_config["value_projection"]
        dv3 = make_model(
            n_speakers, speaker_dim, speaker_embed_std, embed_dim, padding_idx,
            embedding_std, max_positions, n_vocab, freeze_embedding,
            filter_size, encoder_channels, n_mels, decoder_channels, r,
            trainable_positional_encodings, use_memory_mask,
            query_position_rate, key_position_rate, window_backward,
            window_ahead, key_projection, value_projection, downsample_factor,
            linear_dim, use_decoder_states, converter_channels, dropout)

        summary(dv3)

        checkpoint_dir = os.path.join(args.output, "checkpoints")
        if args.checkpoint is not None:
            iteration = io.load_parameters(
                dv3, checkpoint_path=args.checkpoint)
        else:
            iteration = io.load_parameters(
                dv3, checkpoint_dir=checkpoint_dir, iteration=args.iteration)

        # WARNING: don't forget to remove weight norm to re-compute each wrapped layer's weight
        # removing weight norm also speeds up computation
        for layer in dv3.sublayers():
            if isinstance(layer, WeightNormWrapper):
                layer.remove_weight_norm()

        transform_config = config["transform"]
        c = transform_config["replace_pronunciation_prob"]
        sample_rate = transform_config["sample_rate"]
        min_level_db = transform_config["min_level_db"]
        ref_level_db = transform_config["ref_level_db"]
        preemphasis = transform_config["preemphasis"]
        win_length = transform_config["win_length"]
        hop_length = transform_config["hop_length"]

        synthesis_config = config["synthesis"]
        power = synthesis_config["power"]
        n_iter = synthesis_config["n_iter"]

        synthesis_dir = os.path.join(args.output, "synthesis")
        if not os.path.exists(synthesis_dir):
            os.makedirs(synthesis_dir)
        
        mel_only = False
        if args.waveflow_checkpoint_dir:
            with open(args.waveflow_config, 'rt') as f:
                waveflow_config = ruamel.yaml.safe_load(f)
            # fp = open(args.waveflow_config, 'rt')
            # waveflow_config = ruamel.yaml.safe_load(fp)
            waveflow_config['use_fp16'] = False
            waveflow_config['use_gpu'] = False
            
            waveflow_config = Config(**waveflow_config)
            waveflow_model = WaveFlow(waveflow_config,args.waveflow_checkpoint_dir)
            waveflow_iteration = waveflow_model.build()
            mel_only=True

        with open(args.text, "rt", encoding="utf-8") as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                text = line[:-1]
                dv3.eval()

                #Only deepvoice output
                # wav, attn = eval_model(dv3, text, replace_pronounciation_prob,
                #                        min_level_db, ref_level_db, power,
                #                        n_iter, win_length, hop_length,
                #                        preemphasis,mel_only=False)
                # plot_alignment(
                #     attn,
                #     os.path.join(synthesis_dir,
                #                 "before_test_deepvoice3_{}_step_{}.png".format(idx, iteration)))
                # sf.write(
                #     os.path.join(synthesis_dir,
                #                 "before_test_deepvoice3_{}_step{}.wav".format(idx, iteration)),
                #     wav, sample_rate)
    

                wav_mel,linear_outputs, attn = eval_model(dv3, text, replace_pronounciation_prob,
                                       min_level_db, ref_level_db, power,
                                       n_iter, win_length, hop_length,
                                       preemphasis,mel_only=mel_only)
                
                if mel_only:
                    
                    # mel = linear_outputs
                    # print(type(mel), mel.shape)
                    # a,b,c = mel.shape
                    # mel_new=F.reshape(mel,(a,c,b))
                    
                    linear_outputs_np = linear_outputs.numpy()[0].T  # (C, T)
                    
                    denoramlized = np.clip(linear_outputs_np, 0, 1)  * (-min_level_db) + min_level_db
                    lin_scaled = np.exp((denoramlized + ref_level_db) / 20 * np.log(10))
                    
                    #get mel spec
                    mel_filter_bank = librosa.filters.mel(sr=sample_rate,
                                              n_fft=waveflow_config.fft_size,
                                              n_mels=waveflow_config.mel_bands,
                                              fmin=waveflow_config.mel_fmin,
                                              fmax=waveflow_config.mel_fmax)
                    # mel = np.dot(mel_filter_bank, np.abs(lin_scaled)**power)

                    mel = np.dot(mel_filter_bank, np.abs(lin_scaled)**2)
                    # Normalize mel.
                    clip_val = 1e-5
                    ref_constant = 1
                    mel = np.log(np.clip(mel, a_min=clip_val, a_max=None) * ref_constant)
                    # S_mel = librosa.feature.melspectrogram(S=lin_scaled, n_mels=config['transform']['n_mels'], fmin=config['transform']['fmin'], fmax=config['transform']['fmax'], power=1.)
                    
                    # #reshape
                    # a,b=S_mel.shape
                    # S_mel=S_mel.reshape(1,a,b)
                    
                    # #Convert to fluid type
                    # S_mel=dg.to_variable(S_mel)

                    # #pass the mel to waveflow
                    # wav, start_time,syn_time = waveflow_model.infer(S_mel)

                    
                    # max_norm=config['transform']['max_norm']
                    # amplitude_min = np.exp(min_level_db / 20 * np.log(10))  # 1e-5

                    # db scale again
                    # S_mel = 20 * np.log10(np.maximum(amplitude_min,
                                                    # S_mel)) - ref_level_db
                    #Normalize again
                    # S_mel_norm = (S_mel - min_level_db) / (-min_level_db)
                    # S_mel_norm = max_norm * S_mel_norm

                    #clip again to 0,1
                    # if config['transform']['clip_norm']:
                    #     S_mel_norm = np.clip(S_mel_norm, 0, 1)

                    # processed_mel = process_mel(mel_new,waveflow_config)
                    # wav, start_time,syn_time = waveflow_model.infer(processed_mel)

                    # reshape
                    a, b = mel.shape
                    S_mel_norm = mel.reshape(1, a, b)

                    # Convert to fluid type
                    S_mel_norm = dg.to_variable(S_mel_norm)
                    
                    #pass the mel to waveflow
                    wav, start_time,syn_time = waveflow_model.infer(S_mel_norm)

                    wav = wav[0]
                    wav_time = wav.shape[0] / waveflow_config.sample_rate
                    print("audio time {:.4f}, synthesis time {:.4f}".format(wav_time,
                                                                            syn_time))
                    # Denormalize audio from [-1, 1] to [-32768, 32768] int16 range.
                    wav = wav.numpy().astype("float32") * 32768.0
                    wav = wav.astype('int16')
                    sample_rate = waveflow_config.sample_rate  
                else:
                    wav=wav_mel
                plot_alignment(
                    attn,
                    os.path.join(synthesis_dir,
                                 "after_test_{}_step_{}.png".format(idx, iteration)))
                sf.write(
                    os.path.join(synthesis_dir,
                                 "after_test_{}_step{}.wav".format(idx, iteration)),
                    wav, sample_rate) 
                