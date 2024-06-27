import torch
from transformers import AutoModelForAudioClassification, WhisperProcessor, WhisperForConditionalGeneration
from torchinfo import summary
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import os
import shutil
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

'''
Function for training models
'''
# for dataset cleaning

def move_used_files(destination_path, path, csv):

    count = 0
    count_fl = 0
    ls = list(csv['song_name'])
    for x in os.listdir(path):
      if x.lower() in ls:
        count += 1
        shutil.move(os.path.join(path,x), os.path.join(destination_path,x))
        print(f"Moved {x} to {destination_path}")
      else:
        count_fl += 1

    print(f'total of {count} files are kept')
    print(f'total of {count_fl} files are moved')


def separate_data(path, path_save, df, save=True):
    data = []

    explicit_labels = df.set_index('song_name')['explicit'].to_dict()

    for x in os.listdir(path):
        y = x.lower()
        name, ext = os.path.splitext(y)
        song = np.load(os.path.join(path, x))
        save_name = os.path.join(path_save, name)

        if y in explicit_labels:
            label = explicit_labels[y]
        else:
            print(f"song error, check the name: {y}")
            label = None

        print(f'now splitting {x}')
        for i, part in enumerate(song):
            new_name = f"{name}_part_{i}"
            data.append({'song_name': new_name, 'part': i, 'explicit': label})
            np.save(f"{save_name}_part_{i}", part)
            print(f'finished saving {x} part {i}')

    new_df = pd.DataFrame(data)

    if save == True:
        new_df.to_csv(os.path.join(path_save, 'balanced_data_clean_sep.csv'), index=False)
        print('all are saved')
        return new_df
    else:
       return new_df


##############################################################################################################
##############################################################################################################
##############################################################################################################

# Large audio models

class AST(torch.nn.Module):
  def __init__(self, classes, input_dim, conv=True):
    super(AST, self).__init__()
    self.model = AutoModelForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    self.conv = conv
    for param in self.model.parameters():
      param.requires_grad = False
    self.model_no_head = torch.nn.Sequential(*list(self.model.children())[:-1])

    self.head = torch.nn.Sequential(torch.nn.LayerNorm(input_dim), torch.nn.Linear(input_dim, classes))

  def forward(self,x):
    n = self.model_no_head(x)
    x = torch.mean(n[0][:, :2], dim=1) # flatten the dim 1 based on the source code
    y = self.head(x)

    return y

class whisper_enc(torch.nn.Module):
  def __init__(self, classes, input_dim):
    super(whisper_enc, self).__init__()
    self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    for param in self.model.parameters():
      param.requires_grad = False
    self.head = torch.nn.Sequential(torch.nn.LayerNorm(input_dim), torch.nn.Linear(input_dim, classes))

  def forward(self, x):
    encoder_outputs = self.model.model.encoder(x)
    hidden_states = encoder_outputs.last_hidden_state
    logits = self.head(hidden_states[:, 0, :])
    return logits


class dataset(Dataset):
  def __init__(self, dir, df, model='ast'):
    self.dir = dir
    self.df = df
    self.name = df['song_name']
    self.model = model

  def __len__(self):
    return len(self.name)

  def __getitem__(self, idx):
    song_path = os.path.join(self.dir, self.name[idx] + '.npy')
    song = np.load(song_path)

    if self.model == 'crnn':
      song_tensor = torch.tensor(song, dtype=torch.float32)
      song_tensor = torch.einsum('bc->cb', song_tensor)
    else:
       song_tensor = torch.tensor(song, dtype=torch.float32)

    label = self.df['explicit'][idx]
    if label == True:
      label = 1
    else:
      label = 0
    label_tensor = torch.tensor(label, dtype=torch.long)
    return {'arr': song_tensor, 'label': label_tensor}

def optimizer(param, lr=0.001, decay=0):
    return torch.optim.AdamW(param, lr=lr, weight_decay=decay)

def loss_func():
   loss_func = torch.nn.CrossEntropyLoss()
   return loss_func


def train_model(model, loss, optimizer, device, dataset_sizes, dataloders, num_epochs,
              save_path):
  since = time.time()
  loss_all = dict(train=[], test=[])
  accuracy_score = dict(train=[], test=[])

  precisions = dict(train=[], test=[])
  recalls = dict(train=[], test=[])
  f1s = dict(train=[], test=[])

  for epoch in tqdm(range(num_epochs)):

    # Each epoch has a training and validation phase
    for phase in ['train', 'test']:
      if phase == 'train':
          model.train()  # Set model to training mode
      else:
          model.eval()  # Set model to evaluate mode

      running_loss = 0.0
      correct_pred = 0.0
      all_labels = []
      all_preds = []

      # Iterate over data.
      for dicts in tqdm(dataloders[phase], desc=phase):
        sequences = dicts['arr'].to(device)
        label = dicts['label'].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
            logits = model(sequences)

            loss_batch = loss(logits, label)
            # backward + optimize only in training phase
            if phase == 'train':
                loss_batch.backward()
                optimizer.step()

            # statistics
            running_loss += loss_batch.item() * sequences.size(0)

            # predictions
            _, pred = torch.max(logits, 1)
            correct_pred += torch.sum(pred == label.data)

            all_labels.extend(label.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())

      epoch_loss = running_loss / dataset_sizes[phase]
      print(f'loss is: {epoch_loss}')
      loss_all[phase].append(epoch_loss)

      epoch_acc = correct_pred.double() / dataset_sizes[phase]
      print(f'accuracy is:{epoch_acc}')
      accuracy_score[phase].append(epoch_acc.item())


      epoch_precision = precision_score(all_labels, all_preds, average='weighted')
      epoch_recall = recall_score(all_labels, all_preds, average='weighted')
      epoch_f1 = f1_score(all_labels, all_preds, average='weighted')

      print({f'precision is: {epoch_precision}'})
      print(f'recall is: {epoch_recall}')
      print(f'f1 score is: {epoch_f1}')

      precisions[phase].append(epoch_precision)
      recalls[phase].append(epoch_recall)
      f1s[phase].append(epoch_f1)

    print()

  time_elapsed = time.time() - since
  print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

  return model, loss_all, accuracy_score, precisions, recalls, f1s


def visualize_loss_and_acc(loss_all, accuracy_score, save_path):
    df = pd.DataFrame({'epoch': [n for n in range(1,6)],
                    "loss_train": loss_all['train'],
                    "loss_test": loss_all['test'],
                    "acc_train": accuracy_score['train'],
                    "acc_test": accuracy_score['test']})

    df = df.set_index(df['epoch']).drop(['epoch'], axis=1).round(2)

    plt.rcParams["figure.figsize"] = (10,6)

    plt.plot(df['loss_train'], 'b-o', label="Training Loss")
    plt.plot(df['acc_train'], 'r-^', label="Training Accuracy")
    plt.plot(df['loss_test'], 'g-o', label="Test Loss")
    plt.plot(df['acc_test'], 'y-^', label="Test Accuracy")

    # Label the plot.
    plt.title("Training, Test Loss & Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.savefig(os.path.join(save_path, 'loss and acc.png'))


def plot_prf1(p, r,f1, ptest, rtest, f1test, save_path, types='precision recall f1'):
    filename = types + '_report.png'
    plt.rcParams["figure.figsize"] = (10,6)
    plt.plot(p, 'r-', label= 'precision train')
    plt.plot(ptest, 'r--', label= "precision test")
    plt.plot(r, 'y-', label='recall')
    plt.plot(rtest, 'y--', label='recall test')
    plt.plot(f1, 'b-', label= 'f1 train')
    plt.plot(f1test, 'b--', label= 'f1 test')

    # Label the plot.
    plt.title("{} report".format(types))
    plt.xlabel("Epoch")
    plt.ylabel("{}".format(types))
    plt.legend()
    plt.savefig(os.path.join(save_path, filename))
    # plt.show()

    plt.close()


if __name__ == '__main__':

    print(torch.__version__)
    path_csv = r'D:/NLP final project/ast_df_final_sep_1024.csv'
    pth_dataset = r'D:/NLP final project/ast_1024_sep'
    save_path = r'D:/NLP final project/experiments/ast_experiment_1'

    csv = pd.read_csv(path_csv)

    total_samples = 27154
    samples_div = int(27154/2)

    group_True = csv[csv['explicit'] == True]
    group_False = csv[csv['explicit'] == False]

    sample_True = group_True.sample(n=samples_div, random_state=1)
    sample_False = group_False.sample(n=samples_div, random_state=1)

    balanced_df = pd.concat([sample_True, sample_False]).reset_index(drop=True)


    # initiate model
    # model = whisper_enc(classes=2, input_dim=768)
    model =AST(classes=2, input_dim=768)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = model.to(device)
    summary(model, input_size=(8,1024,128))
    
    # loss function and parameters
    loss = loss_func()
    optim  = optimizer(model.parameters(), lr=0.001, decay=0.01)


    # dataset and dataloader
    print('create the dataset....')
    sep_dataset = dataset(dir=pth_dataset,df=balanced_df, model='ast')


    train_ratio = 0.8
    val_ratio = 1 - train_ratio

    num_samples = len(sep_dataset)
    num_train = int(train_ratio * num_samples)
    num_val = num_samples - num_train

    train_set, test_set = random_split(sep_dataset, [num_train, num_val])
    train_dl = DataLoader(train_set, batch_size= 8, num_workers=0, shuffle=True)
    test_dl = DataLoader(test_set, batch_size= 8, num_workers=0, shuffle=False)

    dataset_sizes = {'train': num_train, 'test': num_val}
    dataloders = {'train': train_dl, "test": test_dl}
    num_epochs = 5

    print('train model....')
    model, loss_all, accuracy_score, p, r, f1 = train_model(model, loss, optim, device, dataset_sizes, dataloders, num_epochs,
              save_path)

    visualize_loss_and_acc(loss_all, accuracy_score, save_path)


