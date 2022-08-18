import os
import sys
sys.path.append('/home/ubuntu/Code/TextRecognitionDataGenerator')

import cv2
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import CTCLoss
from torchvision import transforms
import random

from dataset import Synth90kDataset, synth90k_collate_fn
from model import CRNN
from evaluate import evaluate
from config import train_config as config

from trdg.generators import (
    GeneratorFromDict,
    GeneratorFromRandom,
    GeneratorFromStrings,
    GeneratorFromWikipedia,
)

preprocess = transforms.Compose([
    transforms.Resize((100,32),2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



segment_fonts = [
    '/home/ubuntu/Code/fonts-DSEG_v046/DSEG7-Modern-MINI/DSEG7ModernMini-Regular.ttf',
    '/home/ubuntu/Code/fonts-DSEG_v046/DSEG7-Modern-MINI/DSEG7ModernMini-BoldItalic.ttf',
    '/home/ubuntu/Code/fonts-DSEG_v046/DSEG7-Modern-MINI/DSEG7ModernMini-Italic.ttf',
    '/home/ubuntu/Code/fonts-DSEG_v046/DSEG7-Modern-MINI/DSEG7ModernMini-Bold.ttf',
    '/home/ubuntu/Code/fonts-DSEG_v046/DSEG7-Modern-MINI/DSEG7ModernMini-Light.ttf',
    '/home/ubuntu/Code/fonts-DSEG_v046/DSEG7-Modern-MINI/DSEG7ModernMini-LightItalic.ttf',
    '/home/ubuntu/Code/fonts-DSEG_v046/DSEG14-Classic/DSEG14Classic-BoldItalic.ttf',
    '/home/ubuntu/Code/fonts-DSEG_v046/DSEG14-Classic/DSEG14Classic-Light.ttf',
    '/home/ubuntu/Code/fonts-DSEG_v046/DSEG14-Classic/DSEG14Classic-Regular.ttf',
    '/home/ubuntu/Code/fonts-DSEG_v046/DSEG14-Classic/DSEG14Classic-LightItalic.ttf',
    '/home/ubuntu/Code/fonts-DSEG_v046/DSEG14-Classic/DSEG14Classic-Bold.ttf',
    '/home/ubuntu/Code/fonts-DSEG_v046/DSEG14-Classic/DSEG14Classic-Italic.ttf',
    '/home/ubuntu/Code/fonts-DSEG_v046/DSEG14-Modern/DSEG14Modern-Italic.ttf',
    '/home/ubuntu/Code/fonts-DSEG_v046/DSEG14-Modern/DSEG14Modern-Light.ttf',
    '/home/ubuntu/Code/fonts-DSEG_v046/DSEG14-Modern/DSEG14Modern-BoldItalic.ttf',
    '/home/ubuntu/Code/fonts-DSEG_v046/DSEG14-Modern/DSEG14Modern-Regular.ttf',
    '/home/ubuntu/Code/fonts-DSEG_v046/DSEG14-Modern/DSEG14Modern-LightItalic.ttf',
    '/home/ubuntu/Code/fonts-DSEG_v046/DSEG14-Modern/DSEG14Modern-Bold.ttf',
    '/home/ubuntu/Code/fonts-DSEG_v046/DSEG7-Classic-MINI/DSEG7ClassicMini-LightItalic.ttf',
    '/home/ubuntu/Code/fonts-DSEG_v046/DSEG7-Classic-MINI/DSEG7ClassicMini-Regular.ttf',
    '/home/ubuntu/Code/fonts-DSEG_v046/DSEG7-Classic-MINI/DSEG7ClassicMini-BoldItalic.ttf',
    '/home/ubuntu/Code/fonts-DSEG_v046/DSEG7-Classic-MINI/DSEG7ClassicMini-Italic.ttf',
    '/home/ubuntu/Code/fonts-DSEG_v046/DSEG7-Classic-MINI/DSEG7ClassicMini-Light.ttf',
    '/home/ubuntu/Code/fonts-DSEG_v046/DSEG7-Classic-MINI/DSEG7ClassicMini-Bold.ttf',
    '/home/ubuntu/Code/fonts-DSEG_v046/DSEG14-Modern-MINI/DSEG14ModernMini-Italic.ttf',
    '/home/ubuntu/Code/fonts-DSEG_v046/DSEG14-Modern-MINI/DSEG14ModernMini-LightItalic.ttf',
    '/home/ubuntu/Code/fonts-DSEG_v046/DSEG14-Modern-MINI/DSEG14ModernMini-Bold.ttf',
    '/home/ubuntu/Code/fonts-DSEG_v046/DSEG14-Modern-MINI/DSEG14ModernMini-Light.ttf',
    '/home/ubuntu/Code/fonts-DSEG_v046/DSEG14-Modern-MINI/DSEG14ModernMini-Regular.ttf',
    '/home/ubuntu/Code/fonts-DSEG_v046/DSEG14-Modern-MINI/DSEG14ModernMini-BoldItalic.ttf',
    '/home/ubuntu/Code/fonts-DSEG_v046/DSEG7-Classic/DSEG7Classic-Bold.ttf',
    '/home/ubuntu/Code/fonts-DSEG_v046/DSEG7-Classic/DSEG7Classic-Regular.ttf',
    '/home/ubuntu/Code/fonts-DSEG_v046/DSEG7-Classic/DSEG7Classic-Light.ttf',
    '/home/ubuntu/Code/fonts-DSEG_v046/DSEG7-Classic/DSEG7Classic-BoldItalic.ttf',
    '/home/ubuntu/Code/fonts-DSEG_v046/DSEG7-Classic/DSEG7Classic-LightItalic.ttf',
    '/home/ubuntu/Code/fonts-DSEG_v046/DSEG7-Classic/DSEG7Classic-Italic.ttf',
    '/home/ubuntu/Code/fonts-DSEG_v046/DSEG7-7SEGG-CHAN/DSEG7SEGGCHAN-Regular.ttf',
    '/home/ubuntu/Code/fonts-DSEG_v046/DSEG7-7SEGG-CHAN/DSEG7SEGGCHANMINI-Regular.ttf',
    '/home/ubuntu/Code/fonts-DSEG_v046/DSEGWeather/DSEGWeather.ttf',
    '/home/ubuntu/Code/fonts-DSEG_v046/DSEG14-Classic-MINI/DSEG14ClassicMini-Italic.ttf',
    '/home/ubuntu/Code/fonts-DSEG_v046/DSEG14-Classic-MINI/DSEG14ClassicMini-Light.ttf',
    '/home/ubuntu/Code/fonts-DSEG_v046/DSEG14-Classic-MINI/DSEG14ClassicMini-Bold.ttf',
    '/home/ubuntu/Code/fonts-DSEG_v046/DSEG14-Classic-MINI/DSEG14ClassicMini-BoldItalic.ttf',
    '/home/ubuntu/Code/fonts-DSEG_v046/DSEG14-Classic-MINI/DSEG14ClassicMini-Regular.ttf',
    '/home/ubuntu/Code/fonts-DSEG_v046/DSEG14-Classic-MINI/DSEG14ClassicMini-LightItalic.ttf',
    '/home/ubuntu/Code/fonts-DSEG_v046/DSEG7-Modern/DSEG7Modern-Italic.ttf',
    '/home/ubuntu/Code/fonts-DSEG_v046/DSEG7-Modern/DSEG7Modern-BoldItalic.ttf',
    '/home/ubuntu/Code/fonts-DSEG_v046/DSEG7-Modern/DSEG7Modern-Bold.ttf',
    '/home/ubuntu/Code/fonts-DSEG_v046/DSEG7-Modern/DSEG7Modern-Regular.ttf',
    '/home/ubuntu/Code/fonts-DSEG_v046/DSEG7-Modern/DSEG7Modern-LightItalic.ttf',
    '/home/ubuntu/Code/fonts-DSEG_v046/DSEG7-Modern/DSEG7Modern-Light.ttf',
]


class ocr_gen(torch.utils.data.IterableDataset):
    CHARS = '0123456789:.'
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

    def __init__(self):
        self.path = '/home/ubuntu/Code/words.txt'
        # self.count = count
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.generator = GeneratorFromDict(
            path=self.path,
            count=1,
            # length=random.choice([2,3,4,5]),
            length=5,
            fonts= segment_fonts,
            skewing_angle=5,
            random_skew=True,
            blur=1,
            random_blur=True,
            background_type=1,
            text_color='#F8F8F8',
        )
        
        data = self.generator.__next__()
        self.index += 1

        # cv2.imshow('',np.array(data[0]))
        # cv2.waitKey(0)
        
        # cv2.imshow('',preprocess(data[0]).permute((1,2,0)).numpy())
        # cv2.waitKey(0)
        
        # return {'img': preprocess(data[0]), 'idx': self.index-1, 'label': data[1]}
        target = [self.CHAR2LABEL[c] for c in data[1] if c != ' ']

        return preprocess(data[0]).permute(0,2,1), torch.LongTensor(target), torch.LongTensor([len(target)])



    def __len__(self):
        return self.count



def train_batch(crnn, data, optimizer, criterion, device):
    crnn.train()
    images, targets, target_lengths = [d.to(device) for d in data]

    logits = crnn(images)
    log_probs = torch.nn.functional.log_softmax(logits, dim=2)

    batch_size = images.size(0)
    input_lengths = torch.LongTensor([logits.size(0)] * batch_size)
    target_lengths = torch.flatten(target_lengths)
    loss = criterion(log_probs, targets, input_lengths, target_lengths)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(crnn.parameters(), 5) # gradient clipping with 5
    optimizer.step()
    return loss.item()


def main():
    epochs = config['epochs']
    train_batch_size = config['train_batch_size']
    eval_batch_size = config['eval_batch_size']
    lr = config['lr']
    show_interval = config['show_interval']
    valid_interval = config['valid_interval']
    save_interval = config['save_interval']
    cpu_workers = config['cpu_workers']
    reload_checkpoint = config['reload_checkpoint']
    valid_max_iter = config['valid_max_iter']

    img_width = config['img_width']
    img_height = config['img_height']
    data_dir = config['data_dir']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    # train_dataset = Synth90kDataset(root_dir=data_dir, mode='train',
    #                                 img_height=img_height, img_width=img_width)
    # valid_dataset = Synth90kDataset(root_dir=data_dir, mode='dev',
    #                                 img_height=img_height, img_width=img_width)

    train_dataset = ocr_gen()
    valid_dataset = ocr_gen()

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        # shuffle=True,
        num_workers=cpu_workers,)
        # collate_fn=synth90k_collate_fn)
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=eval_batch_size,
        # shuffle=True,
        num_workers=cpu_workers,)
        # collate_fn=synth90k_collate_fn)

    # num_class = len(Synth90kDataset.LABEL2CHAR) + 1
    num_class = 12
    crnn = CRNN(3, img_height, img_width, num_class,
                map_to_seq_hidden=config['map_to_seq_hidden'],
                rnn_hidden=config['rnn_hidden'],
                leaky_relu=config['leaky_relu'])
    if reload_checkpoint:
        crnn.load_state_dict(torch.load(reload_checkpoint, map_location=device))
    crnn.to(device)

    optimizer = optim.RMSprop(crnn.parameters(), lr=lr)
    criterion = CTCLoss(reduction='sum', zero_infinity=True)
    criterion.to(device)

    assert save_interval % valid_interval == 0
    i = 1
    for epoch in range(1, epochs + 1):
        print(f'epoch: {epoch}')
        tot_train_loss = 0.
        tot_train_count = 0
        for train_data in train_loader:
            loss = train_batch(crnn, train_data, optimizer, criterion, device)
            train_size = train_data[0].size(0)

            tot_train_loss += loss
            tot_train_count += train_size
            if i % show_interval == 0:
                print('train_batch_loss[', i, ']: ', loss / train_size)

            if i % valid_interval == 0:
                evaluation = evaluate(crnn, valid_loader, criterion,
                                      max_iter=config['valid_max_iter'],
                                      decode_method=config['decode_method'],
                                      beam_size=config['beam_size'])
                print('valid_evaluation: loss={loss}, acc={acc}'.format(**evaluation))

                if i % save_interval == 0:
                    prefix = 'crnn'
                    loss = evaluation['loss']
                    save_model_path = os.path.join(config['checkpoints_dir'],
                                                   f'{prefix}_{i:06}_loss{loss}.pt')
                    torch.save(crnn.state_dict(), save_model_path)
                    print('save model at ', save_model_path)

            i += 1

        print('train_loss: ', tot_train_loss / tot_train_count)


if __name__ == '__main__':
    main()
