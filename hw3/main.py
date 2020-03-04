# coding: utf-8
import argparse
import time
import math
import torch
import torch.nn as nn
import data
import model
import bleu
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch ptb Language Model')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--train_batch_size', type=int, default=128, metavar='N',
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=128, metavar='N',
                    help='eval batch size')
parser.add_argument('--max_sql', type=int, default=35,
                    help='sequence length')
parser.add_argument('--seed', type=int, default=1234,
                    help='set random seed')
parser.add_argument('--gpu_id', type=int, default=-1, help='GPU device id used')
parser.add_argument('--emsize', type=int, default=512, help='size of word embeddings ')
parser.add_argument('--nhid', type=int, default=512,
                    help='number of hidden units per layer ')
parser.add_argument('--nlayers', type=int, default=1, help='number of layers ')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights ')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--log_interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--truncated_bptt_step', type=int, default=20, metavar='N',
                    help='step at which it truncates bptt (default: 20)')

args = parser.parse_args()

writer = SummaryWriter(log_dir="summary/{}".format(args.model))

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

# Use gpu or cpu to train
use_gpu = True

if args.gpu_id > -1:
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(args.gpu_id)
else:
    device = torch.device("cpu")

# load data
train_batch_size = args.train_batch_size
eval_batch_size = args.eval_batch_size
batch_size = {'train': train_batch_size, 'valid': eval_batch_size}
data_loader = data.Corpus("data/ptb", batch_size, args.max_sql)

# WRITE CODE HERE within two '#' bar
########################################
# Build LMModel model (bulid your language model here)
nvoc = len(data_loader.vocabulary)
# model = model.LMModel(rnn_type=args.model,
#                       nvoc=nvoc,
#                       ninput=args.emsize,
#                       nhid=args.nhid,
#                       nlayers=args.nlayers,
#                       tie_weights=args.tied)
model = model.LSTMAtt(rnn_type=args.model, nvoc=nvoc, ninput=args.emsize, nhid=args.nhid, nlayers=args.nlayers)

#使用LN版本，需要修改107行和134行 hidden = None
# model = model.LSTM_LN_Att(rnn_type=args.model, nvoc=nvoc, ninput=args.emsize, nhid=args.nhid, nlayers=args.nlayers)
model = model.to(device)
lr = args.lr
best_val_loss = None
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

criterion = nn.CrossEntropyLoss()


########################################
# Evaluation Function
# Calculate the average cross-entropy loss between the prediction and the ground truth word.
# And then exp(average cross-entropy loss) is perplexity.

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def evaluate():
    model.eval()
    data_loader.set_valid()
    total_loss = 0.0
    bleu_sum = 0.0
    nvoc = len(data_loader.word_id)
    hidden = model.init_hidden(eval_batch_size)
    # hidden = None
    with torch.no_grad():
        for i in range(0, data_loader.valid.size(0), args.max_sql):
            data, target, end_flag = data_loader.get_batch()
            data = data.to(device)
            target = target.to(device)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, nvoc)
            total_loss += len(data) * criterion(output_flat, target).item()
            hidden = repackage_hidden(hidden)
            _, predicted = torch.max(output, -1)
            label = target.view(predicted.size()).data.cpu().numpy()
            predicted = predicted.data.cpu().numpy()
            bleu_sum += len(data) * bleu.cal_blue_score(predicted, label)
    return total_loss / (len(data_loader.valid)), bleu_sum/(len(data_loader.valid))



# Train Function
def train():
    model.train()
    data_loader.set_train()
    end_flag = False
    total_loss = 0.0
    epoch_loss = 0.0
    start_time = time.time()
    hidden = model.init_hidden(eval_batch_size)
    # hidden = None

    batch = 0
    while not end_flag:
        data, target, end_flag = data_loader.get_batch()
        hidden = repackage_hidden(hidden)
        data = data.to(device)
        target = target.to(device)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, nvoc), target)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()

        total_loss += loss.item()
        # epoch_loss = total_loss / nvoc
        epoch_loss = total_loss / (data_loader.train.size(0) / args.max_sql)


        batch += 1
    print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:03.3f} | time {:5.2f} | '
          'loss {:5.2f} | ppl {:8.2f}'.format(
        epoch, batch, data_loader.train_batch_num, lr,
        time.time() - start_time, epoch_loss, math.pow(2, epoch_loss)))
    writer.add_scalar("Train/loss", epoch_loss, epoch)
    writer.add_scalar("Train/ppl", math.pow(2, epoch_loss), epoch)

########################################


# Loop over epochs.
for epoch in range(1, args.epochs + 1):
    epoch_start_time = time.time()
    train()
    val_loss, bleu_sum = evaluate()
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:6.2f} | bleu {:3.2f} |'.format(epoch, (time.time() - epoch_start_time), val_loss,
                                                      math.pow(2, val_loss), bleu_sum))
    print('-' * 89)
    # Save the model if the validation loss is the best we've seen so far.
    if not best_val_loss or val_loss < best_val_loss:
        with open(args.save, 'wb') as f:
            torch.save(model, f)
        best_val_loss = val_loss
    # else:
    #     # Anneal the learning rate if no improvement has been seen in the validation dataset.
    #     lr /= 4.0
    writer.add_scalar("Val/loss", val_loss, epoch)
    writer.add_scalar("Val/ppl", math.pow(2, val_loss), epoch)
    writer.add_scalar("Val/BLEU", bleu_sum, epoch)
print('| The end | best valid loss {:5.2f} | best valid pp {:8.2f}'.format(best_val_loss, math.pow(2, best_val_loss)))
