import argparse
import torch
import time
import json
import numpy as np
import math
import random

np.random.seed(1337)
random.seed(1337)
torch.manual_seed(1337)
torch.cuda.manual_seed(1337)

def batch_generator(X, y, batch_size=128, return_idx=False, crf=False):
    for offset in range(0, X.shape[0], batch_size):
        batch_X_len=np.sum(X[offset:offset+batch_size]!=0, axis=1)
        batch_idx=batch_X_len.argsort()[::-1]
        batch_X_len=batch_X_len[batch_idx]
        batch_X_mask=(X[offset:offset+batch_size]!=0)[batch_idx].astype(np.uint8)
        batch_X=X[offset:offset+batch_size][batch_idx] 
        batch_y=y[offset:offset+batch_size][batch_idx]
        batch_X = torch.autograd.Variable(torch.from_numpy(batch_X).long().cuda() )
        batch_X_mask=torch.autograd.Variable(torch.from_numpy(batch_X_mask).long().cuda() )
        batch_y = torch.autograd.Variable(torch.from_numpy(batch_y).long().cuda() )
        if len(batch_y.size() )==2 and not crf:
            batch_y=torch.nn.utils.rnn.pack_padded_sequence(batch_y, batch_X_len, batch_first=True)
        if return_idx: #in testing, need to sort back.
            yield (batch_X, batch_y, batch_X_len, batch_X_mask, batch_idx)
        else:
            yield (batch_X, batch_y, batch_X_len, batch_X_mask)
            
class Model(torch.nn.Module):
    def __init__(self, gen_emb, domain_emb, num_classes=3, dropout=0.5, crf=False):
        super(Model, self).__init__()
        self.gen_embedding = torch.nn.Embedding(gen_emb.shape[0], gen_emb.shape[1])
        self.gen_embedding.weight = torch.nn.Parameter(torch.from_numpy(gen_emb), requires_grad=False)
        self.domain_embedding = torch.nn.Embedding(domain_emb.shape[0], domain_emb.shape[1])
        self.domain_embedding.weight = torch.nn.Parameter(torch.from_numpy(domain_emb), requires_grad=False)

        self.conv1 = torch.nn.Conv1d(gen_emb.shape[1] + domain_emb.shape[1], 128, 5, padding=2)
        self.conv2 = torch.nn.Conv1d(gen_emb.shape[1] + domain_emb.shape[1], 128, 3, padding=1)
        self.dropout = torch.nn.Dropout(dropout)

        self.conv3 = torch.nn.Conv1d(256, 256, 5, padding=2)
        self.conv4 = torch.nn.Conv1d(256, 256, 5, padding=2)
        self.conv5 = torch.nn.Conv1d(256, 256, 5, padding=2)
        self.linear_ae = torch.nn.Linear(512, num_classes)

        self._conv1 = torch.nn.Conv1d(gen_emb.shape[1] + domain_emb.shape[1], 128, 5, padding=2)
        self._conv2 = torch.nn.Conv1d(gen_emb.shape[1] + domain_emb.shape[1], 128, 3, padding=1)
        self._dropout = torch.nn.Dropout(dropout)

        self._fc1 = torch.nn.Linear(256, 256, bias=False)
        self._fc2 = torch.nn.Linear(256, 256, bias=False)

        self._conv3 = torch.nn.Conv1d(256, 256, 5, padding=2)
        self._conv4 = torch.nn.Conv1d(256, 256, 5, padding=2)
        self._conv5 = torch.nn.Conv1d(256, 256, 5, padding=2)
        self._linear_ae = torch.nn.Linear(256, num_classes)
        self.crf_flag = crf
        if self.crf_flag:
            from allennlp.modules import ConditionalRandomField
            self.crf = ConditionalRandomField(num_classes)

    def forward(self, x, x_len, x_mask, x_tag=None, testing=False):
        x_emb = torch.cat((self.gen_embedding(x), self.domain_embedding(x)), dim=2)
        # print('初始emb维度:', x_emb.shape)
        x_emb = self.dropout(x_emb).transpose(1, 2)

        op_conv = x_emb

        # print('emb维度:',x_emb.shape)
        x_conv = torch.nn.functional.relu(torch.cat((self.conv1(x_emb), self.conv2(x_emb)), dim=1))
        # print('cov1维度:', x_conv.shape)
        x_conv = self.dropout(x_conv)
        x_conv = torch.nn.functional.relu(self.conv3(x_conv))
        # print('cov3维度:', x_conv.shape)
        x_conv = self.dropout(x_conv)
        x_conv = torch.nn.functional.relu(self.conv4(x_conv))
        # print('cov4维度:', x_conv.shape)
        x_conv = self.dropout(x_conv)
        x_conv = torch.nn.functional.relu(self.conv5(x_conv))
        # print('cov5维度:', x_conv.shape)
        as_conv = x_conv
        '''
        as_input=self.dropout(x_conv).transpose(1, 2)
        #print('*:', as_input.shape)

        op_input = x_emb1.transpose(1, 2)
        #print('**:', op_input.shape)
        as_input=self._fc1(as_input)
        op_input=self._fc2(op_input)
        #print('as_input:', as_input.shape)
        #print('op_input维度:', op_input.shape)
        #print('as+op_input维度:', torch.nn.functional.relu(as_input+op_input).shape)
        op_conv=x_emb1.transpose(1, 2)+torch.nn.functional.relu(as_input+op_input)

        op_conv = self.dropout(op_conv).transpose(1, 2)
        '''
        op_conv = torch.nn.functional.relu(torch.cat((self._conv1(op_conv), self._conv2(op_conv)), dim=1))
        # print('cov1维度:', op_conv.shape)
        op_conv = self.dropout(op_conv)
        op_conv = torch.nn.functional.relu(self._conv3(op_conv))
        # print('cov3维度:', x_conv.shape)
        op_conv = self.dropout(op_conv)
        op_conv = torch.nn.functional.relu(self._conv4(op_conv))
        # print('cov4维度:', x_conv.shape)
        op_conv = self.dropout(op_conv)
        op_conv = torch.nn.functional.relu(self._conv5(op_conv))
        # print('op_conv维度:', op_conv.shape)

        as_conv = as_conv.transpose(1, 2)
        op_conv = op_conv.transpose(1, 2)
        # print('!as_conv维度:', as_conv.shape)
        # print('!op_conv维度:', op_conv.shape)
        #x_logit_opi = self._linear_ae(op_conv)
        as_conv = self._fc1(as_conv)
        # print('as_conv维度:', as_conv.shape)
        op_conv = self._fc2(op_conv)
        # print('op_conv维度:', op_conv.shape)
        op_conv = op_conv + torch.nn.functional.relu(as_conv + op_conv)

        as_conv = as_conv.transpose(1, 2)
        op_conv = op_conv.transpose(1, 2)

        ans_conv = torch.cat((as_conv, op_conv), dim=1)
        # print('ans_conv维度:', ans_conv.shape)

        x_conv = ans_conv.transpose(1, 2)
        # print('transpose维度:', x_conv.shape)
        x_logit = self.linear_ae(x_conv)
        # print('linear维度:', x_logit.shape)
        if testing:
            if self.crf_flag:
                score=self.crf.viterbi_tags(x_logit, x_mask)
            else:
                x_logit=x_logit.transpose(2, 0)
                score=torch.nn.functional.log_softmax(x_logit).transpose(2, 0)
        else:
            if self.crf_flag:
                score=-self.crf(x_logit, x_tag, x_mask)
            else:
                x_logit=torch.nn.utils.rnn.pack_padded_sequence(x_logit, x_len, batch_first=True)
                score=torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(x_logit.data), x_tag.data)
        return score

def valid_loss(model, valid_X, valid_y, crf=False):
    model.eval()
    losses=[]
    for batch in batch_generator(valid_X, valid_y, crf=crf):
        batch_valid_X, batch_valid_y, batch_valid_X_len, batch_valid_X_mask=batch
        loss=model(batch_valid_X, batch_valid_X_len, batch_valid_X_mask, batch_valid_y)
        losses.append(loss.data[0])
    model.train()
    return sum(losses)/len(losses)

def train(train_X, train_y, valid_X, valid_y, model, model_fn, optimizer, parameters, epochs=200, batch_size=128, crf=False):
    best_loss=float("inf") 
    valid_history=[]
    train_history=[]
    for epoch in range(epochs):
        print('epoch',epoch,':', end='')
        for batch in batch_generator(train_X, train_y, batch_size, crf=crf):
            #print(batch,' ', end='')
            batch_train_X, batch_train_y, batch_train_X_len, batch_train_X_mask=batch
            loss=model(batch_train_X, batch_train_X_len, batch_train_X_mask, batch_train_y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(parameters, 1.)
            optimizer.step()
        loss=valid_loss(model, train_X, train_y, crf=crf)
        print("loss%d : %f" % (epoch,loss))
        train_history.append(loss)
        loss=valid_loss(model, valid_X, valid_y, crf=crf)
        valid_history.append(loss)        
        if loss<best_loss:
            best_loss=loss
            torch.save(model, model_fn)
            print("update")
        shuffle_idx=np.random.permutation(len(train_X) )
        train_X=train_X[shuffle_idx]
        train_y=train_y[shuffle_idx]
    model=torch.load(model_fn) 
    return train_history, valid_history

def run(domain, data_dir, model_dir, valid_split, runs, epochs, lr, dropout, batch_size=128):
    gen_emb=np.load(data_dir+"gen.vec.npy")
    domain_emb=np.load(data_dir+domain+"_emb.vec.npy")

    ae_data=np.load(data_dir+domain+".npz")
    
    valid_X=ae_data['train_X'][-valid_split:]
    valid_y=ae_data['train_y'][-valid_split:]
    train_X=ae_data['train_X'][:-valid_split]
    train_y=ae_data['train_y'][:-valid_split]
    '''
    ae_data = np.load(data_dir + domain + "_data.npz")

    # print("gen_emb:", gen_emb.shape)
    # print("domain_emb:", domain_emb.shape)

    valid_X = ae_data['sentences'][-valid_split:]
    valid_y = ae_data['aspect_tags'][-valid_split:]
    valid_y_opi = ae_data['opinion_tags'][-valid_split:]
    train_X = ae_data['sentences'][:-valid_split]
    train_y = ae_data['aspect_tags'][:-valid_split]
    train_y_opi = ae_data['opinion_tags'][:-valid_split]
    '''
    for r in range(runs):
        print(r)
        model=Model(gen_emb, domain_emb, 3, dropout=dropout, crf=False)
        model.cuda()
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.Adam(parameters, lr=lr)
        train_history, valid_history=train(train_X, train_y, valid_X, valid_y, model, model_dir+domain+str(r), optimizer, parameters, epochs, crf=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default="model/")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200) 
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--domain', type=str, default="laptop")
    parser.add_argument('--data_dir', type=str, default="data/prep_data/")
    parser.add_argument('--valid', type=int, default=150) #number of validation data.
    parser.add_argument('--lr', type=float, default=0.0001) 
    parser.add_argument('--dropout', type=float, default=0.55) 

    args = parser.parse_args()

    run(args.domain, args.data_dir, args.model_dir, args.valid, args.runs, args.epochs, args.lr, args.dropout, args.batch_size)

