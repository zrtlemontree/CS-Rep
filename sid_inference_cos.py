import os
import argparse
import torch
import kaldiio
import soundfile as sf
import torch.nn as nn
from rep_tdnn import Xvector
from feature_extractor import CMVN, spectrogram
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict



'''
egs:
--trials trials --checkpoint trained_model/net.pth   --feature-path /data1/data_fbank/voxceleb1/test  --feature-type wav --gpu-id "0" --output ./
'''


def read_keyvalue_file_aslist(file_path):
    '''
    used to read utt2spk feat.scp wav.scp
    :param file_path:
    :return:
    '''
    dic = []
    with open(file_path, mode='r') as f:
        for line in f:
            items = line.strip().split(' ')
            key = items[0]
            value = items[1]
            dic.append([key, value])
    return dic



class SpeakerTestWaveDataset(Dataset):
    '''
    SpeakerTestWaveDataset
    '''
    def __init__(self, feats_scp):
        self.feats_scp_list = feats_scp
        self.count = len(self.feats_scp_list)
        self.feats_list = self.feats_scp_list


    def __len__(self):
        return self.count

    def __getitem__(self, sid):

        wav, sr = self.load_audio(self.feats_list[sid][1])
        # [N, T]
        feature = spectrogram(wav)
        # [N, T]
        #print(feature.shape)
        feature = CMVN(feature)
        feature = feature.transpose()
        return self.feats_list[sid][0], feature

    @staticmethod
    def load_audio(filename, start=0, stop=None):
        y, sr = sf.read(filename, start=start, stop=stop, dtype='float32', always_2d=True)
        y = y[:, 0]
        return y, sr


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, help="The path of model")
    parser.add_argument("--trials", type=str, help="The path of feature")
    parser.add_argument("--feature-path", type=str, help="The path of feature")
    parser.add_argument("--model-type", type=str, default='torch', help="The model type of inference, including torch, onnx")
    parser.add_argument("--feature-type", type=str, default='fbank', help="The feature type of inference, including fbank, wav")
    parser.add_argument("--gpu-id", type=str, default=0, help="The id of gpu processing.")
    parser.add_argument("--output", type=str, help="The path of output embeddings.")
    parser.add_argument("--rep", type=str, help="Used CS-rep or not.")

    return parser.parse_args()


def generate_model(args):
    n_classes = 5994
    embedding_size = 512
    model = Xvector(n_classes=n_classes, embedding_size=embedding_size, m=0, s=0)
    return model


def load_model(args, model, device):
    '''
    load params of model

    :param args:
    :param model:
    :return:
    '''
    resume = args.checkpoint
    model_dir = resume
    if os.path.isfile(model_dir):

        print('=> loading checkpoint {}'.format(model_dir))

        # original saved file with DataParallel
        state_dict = torch.load(model_dir, map_location='cpu')

        new_state_dict = OrderedDict()
        #for k, v in state_dict['state_dict'].items():
        for k, v in state_dict.items():
            #print(k)
            name = k[7:]  # remove 'module'
            new_state_dict[name] = v

        state_dict = new_state_dict

        # load params
        model.load_state_dict(state_dict)


    model.eval()
    print(args.rep)
    if args.rep == "True":
        print("CS-Rep was used!!")
        model.embedding_net.rep_all()

    import torchsummary
    torchsummary.summary(model, (161, 300),device='cpu')
    model.to(device)  # move model to GPU
    model = nn.DataParallel(model)
    return model


def building_torch_dataset(args, wavscp,):
    '''
    building_torch_dataset


    :param args:
    :param wavscp:
    :return:
    '''
    return SpeakerTestWaveDataset(wavscp)



def inference(args, model, test_dataloader, device, wspecifier):
    '''
    inference and save embedding to ark

    :param args:
    :param model:
    :param test_dataloader:
    :param device:
    :param wspecifier:
    :return:
    '''
    model.eval()
    with kaldiio.WriteHelper(wspecifier) as writer:
        for batch_idx, (utt_id, feature) in enumerate(test_dataloader):
            with torch.no_grad():
                feature = feature.to(device)
                _, embedding = model.forward(feature)
                embedding = embedding.data.cpu().numpy()
                utt_id = utt_id[0]
                writer(utt_id, embedding[0])
    writer.close()


def extract_embedding(args, gpu_id, feats_scp_list):
    '''
    extract embedding for one worker
    :param args:
    :param gpu_id:
    :param feats_scp_list:
    :return:
    '''
    if args.feature_type == 'wav':

        # 1. set gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        device = torch.device('cuda')

        # 2. load model
        model = generate_model(args)
        model = load_model(args, model, device)

        # 3. dataset  & dataloader
        dataset = building_torch_dataset(args, feats_scp_list)
        test_loader = DataLoader(dataset, batch_size=1, shuffle=False,
                                 num_workers=16, pin_memory=True)
        # 4. inference get embedding
        wspecifier = "ark,scp:{}/xvectors.{}.ark,{}/xvectors.{}.scp".format(args.output, gpu_id, args.output, gpu_id)
        inference(args, model, test_loader, device, wspecifier)


def kaldi_cos(args):
    kaldi_cmd = '''
    ivector-compute-dot-products "cat {trials}|cut -d\  --fields=1,2 |"\
     "ark:copy-vector scp:{xv_path}/xvectors.scp ark:- |ivector-normalize-length ark:- ark:- |"\
      "ark:copy-vector scp:{xv_path}/xvectors.scp ark:- |ivector-normalize-length ark:- ark:- |"  {xv_path}/cos_score.score
    '''.format(
        trials=args.trials, xv_path=args.output
    )
    os.system(kaldi_cmd)

    cos_cmd = '''
    paste -d  ' ' {trials} {xv_path}/cos_score.score | awk -F ' ' '{{print$6,$3}}'| compute-eer  - 
    '''.format(
        trials=args.trials, xv_path=args.output
    )
    os.system(cos_cmd)

    cos_cmd = '''
        sid/compute_min_dcf.py --c-miss 10  --p-target 0.01  {xv_path}/cos_score.score {trials}
        '''.format(
        trials=args.trials, xv_path=args.output
    )
    os.system(cos_cmd)

    cos_cmd = '''
            sid/compute_min_dcf.py --p-target 0.001 {xv_path}/cos_score.score {trials}
            '''.format(
        trials=args.trials, xv_path=args.output
    )
    os.system(cos_cmd)




def main():
    args = parse_args()

    feats_scp_list = read_keyvalue_file_aslist(os.path.join(args.feature_path, "wav.scp"))
    args.gpu_id = eval(args.gpu_id)

    extract_embedding(args, "{}".format(args.gpu_id), feats_scp_list)

    # xv.ark
    linux_cp_cmd = "cat {output}/xvectors.{id}.scp > {output}/xvectors.scp".format(output=args.output, id=args.gpu_id)
    os.system(linux_cp_cmd)
    print(linux_cp_cmd)

    # scoring
    kaldi_cos(args)


if __name__ == "__main__":
    main()



