import argparse
import torch
import torch.nn.functional as F
import time
import json
import numpy as np
import math
import random

np.random.seed(1337)
random.seed(1337)
torch.manual_seed(1337)
torch.cuda.manual_seed(1337)

def batch_generator(X, y, y_opi, batch_size=128, return_idx=False, crf=False):
    for offset in range(0, X.shape[0], batch_size):
        batch_X_len=np.sum(X[offset:offset+batch_size]!=0, axis=1)
        batch_idx=batch_X_len.argsort()[::-1]
        batch_X_len=batch_X_len[batch_idx]
        batch_X_mask=(X[offset:offset+batch_size]!=0)[batch_idx].astype(np.uint8)
        batch_X=X[offset:offset+batch_size][batch_idx] 
        batch_y=y[offset:offset+batch_size][batch_idx]
        batch_y_opi = y_opi[offset:offset + batch_size][batch_idx]
        batch_X = torch.autograd.Variable(torch.from_numpy(batch_X).long().cuda() )
        batch_X_mask=torch.autograd.Variable(torch.from_numpy(batch_X_mask).long().cuda() )
        batch_y = torch.autograd.Variable(torch.from_numpy(batch_y).long().cuda() )
        batch_y_opi = torch.autograd.Variable(torch.from_numpy(batch_y_opi).long().cuda())
        if len(batch_y.size() )==2 and not crf:
            batch_y=torch.nn.utils.rnn.pack_padded_sequence(batch_y, batch_X_len, batch_first=True)
        if len(batch_y_opi.size() )==2 and not crf:
            batch_y_opi=torch.nn.utils.rnn.pack_padded_sequence(batch_y_opi, batch_X_len, batch_first=True)
        if return_idx: #in testing, need to sort back.
            yield (batch_X, batch_y, batch_y_opi, batch_X_len, batch_X_mask, batch_idx)
        else:
            yield (batch_X, batch_y, batch_y_opi, batch_X_len, batch_X_mask)
            
class Model(torch.nn.Module):
    def __init__(self, gen_emb, domain_emb, num_classes=3, dropout=0.6, crf=False):
        super(Model, self).__init__()
        self.gen_embedding = torch.nn.Embedding(gen_emb.shape[0], gen_emb.shape[1])
        self.gen_embedding.weight=torch.nn.Parameter(torch.from_numpy(gen_emb), requires_grad=False)
        self.domain_embedding = torch.nn.Embedding(domain_emb.shape[0], domain_emb.shape[1])
        self.domain_embedding.weight=torch.nn.Parameter(torch.from_numpy(domain_emb), requires_grad=False)
    
        self.conv1=torch.nn.Conv1d(gen_emb.shape[1]+domain_emb.shape[1], 128, 5, padding=2 )
        self.conv2=torch.nn.Conv1d(gen_emb.shape[1]+domain_emb.shape[1], 128, 3, padding=1 )
        self.dropout=torch.nn.Dropout(dropout)

        self.dropout2 = torch.nn.Dropout(dropout)

        self.conv3=torch.nn.Conv1d(256, 256, 5, padding=2)
        self.conv4=torch.nn.Conv1d(256, 256, 5, padding=2)
        self.conv5=torch.nn.Conv1d(256, 256, 5, padding=2)
        self.linear_ae=torch.nn.Linear(512, num_classes)


        self._conv1 = torch.nn.Conv1d(gen_emb.shape[1] + domain_emb.shape[1], 128, 5, padding=2)
        self._conv2 = torch.nn.Conv1d(gen_emb.shape[1] + domain_emb.shape[1], 128, 3, padding=1)
        self._dropout = torch.nn.Dropout(dropout)

        self._fc1=torch.nn.Linear(256, 256, bias=False)
        self._fc2 = torch.nn.Linear(256, 256, bias=False)

        self._conv3 = torch.nn.Conv1d(256, 256, 5, padding=2)
        self._conv4 = torch.nn.Conv1d(256, 256, 5, padding=2)
        self._conv5 = torch.nn.Conv1d(256, 256, 5, padding=2)
        self._linear_ae = torch.nn.Linear(256, 2)
        self.fc3 = torch.nn.Linear(166, 83)
        self.atten1=torch.nn.Linear(166, 83)
        self.atten=torch.nn.Linear(256, 256,bias=False)
        self.atten_w=torch.nn.Linear(83,1)

        self.crf_flag=crf
        if self.crf_flag:
            from allennlp.modules import ConditionalRandomField
            self.crf=ConditionalRandomField(num_classes)            
          
    def forward(self, x, x_len, x_mask, x_tag=None, x_tag_opi=None, testing=False):
        x_emb=torch.cat((self.gen_embedding(x), self.domain_embedding(x) ), dim=2)
        #print('初始emb维度:', x_emb.shape)
        op_conv=x_emb
        x_emb=self.dropout(x_emb).transpose(1, 2)




        #print('emb维度:',x_emb.shape)
        x_conv=torch.nn.functional.relu(torch.cat((self.conv1(x_emb), self.conv2(x_emb)), dim=1) )
        #print('cov1维度:', x_conv.shape)
        x_conv=self.dropout(x_conv)
        x_conv=torch.nn.functional.relu(self.conv3(x_conv) )
        #print('cov3维度:', x_conv.shape)
        x_conv=self.dropout(x_conv)
        x_conv=torch.nn.functional.relu(self.conv4(x_conv) )
        #print('cov4维度:', x_conv.shape)
        x_conv=self.dropout(x_conv)
        x_conv=torch.nn.functional.relu(self.conv5(x_conv) )
        #print('cov5维度:', x_conv.shape)
        as_conv=x_conv
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
        op_conv = self.dropout2(op_conv).transpose(1, 2)
        op_conv = torch.nn.functional.relu(torch.cat((self._conv1(op_conv), self._conv2(op_conv)), dim=1))
        #print('cov1维度:', op_conv.shape)
        op_conv = self.dropout2(op_conv)
        op_conv = torch.nn.functional.relu(self._conv3(op_conv))
        #print('cov3维度:', x_conv.shape)
        op_conv = self.dropout2(op_conv)
        op_conv = torch.nn.functional.relu(self._conv4(op_conv))
        #print('cov4维度:', x_conv.shape)
        op_conv = self.dropout2(op_conv)
        op_conv = torch.nn.functional.relu(self._conv5(op_conv))
        #print('op_conv维度:', op_conv.shape)

        as_conv = as_conv.transpose(1, 2)#[128, 83, 256]
        op_conv = op_conv.transpose(1, 2)#[128, 83, 256]
        #print('!as_conv维度:', as_conv.shape)
        #print('!op_conv维度:', op_conv.shape)


        '''
        as_conv = self._fc1(as_conv)
        #print('as_conv维度:', as_conv.shape)
        op_conv = self._fc2(op_conv)
        #print('op_conv维度:', op_conv.shape)
        op_conv = op_conv + torch.nn.functional.relu(as_conv + op_conv) #[128, 83, 256]
        '''
        x_logit_opi = self._linear_ae(op_conv)

        atten=self.atten(op_conv)#[128, 83, 256]
        atten=torch.bmm(as_conv,atten.transpose(1, 2))#[128, 83, 256]
        #atten=self.atten_w(atten)

        atten_weight=F.softmax(F.relu(atten),dim=1)
        #print('attenw维度:', atten_weight)
        atten_conv=torch.bmm(op_conv.transpose(1, 2),atten_weight.transpose(1, 2))#[128, 256, 83]
        ans_conv = torch.cat((as_conv, atten_conv.transpose(1, 2)), dim=2)
        #print('ans_conv维度:', ans_conv.shape)
        '''


        as_conv = as_conv.transpose(1, 2)
        op_conv = op_conv.transpose(1, 2) #[128, 256, 83]

        ans_conv=torch.cat((as_conv,op_conv), dim=2)#[128, 256, 166]

        atten=self.atten1(ans_conv)#[128, 256, 83]
        #print('atten维度:', atten.shape)

        ans_conv=torch.cat((as_conv,atten), dim=2)#[128, 256, 166]
        ans_conv=self.fc3(ans_conv)
        '''
        #x_conv=atten.transpose(1, 2)

        #print('transpose维度:', x_conv.shape)
        x_logit=self.linear_ae(ans_conv)
        #print('linear维度:', x_logit.shape)
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
                score1=torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(x_logit.data), x_tag.data)

                x_logit_opi = torch.nn.utils.rnn.pack_padded_sequence(x_logit_opi, x_len, batch_first=True)
                score2 = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(x_logit_opi.data), x_tag_opi.data)
                #print('aspect_loss:',score1,'opinion_loss:',score2)
        return score1,score2

def valid_loss(model, valid_X, valid_y,valid_y_opi, crf=False):
    model.eval()
    losses1=[]
    losses2 = []
    for batch in batch_generator(valid_X, valid_y,valid_y_opi, crf=crf):
        batch_valid_X, batch_valid_y,batch_valid_y_opi, batch_valid_X_len, batch_valid_X_mask=batch
        loss1,loss2=model(batch_valid_X, batch_valid_X_len, batch_valid_X_mask, batch_valid_y,batch_valid_y_opi)
        losses1.append(loss1.data[0])
        losses2.append(loss2.data[0])
    model.train()
    return sum(losses1)/len(losses1),sum(losses2)/len(losses2)

def train(train_X, train_y, train_y_opi, valid_X, valid_y, valid_y_opi, model, model_fn, optimizer, parameters,optimizer2, parameters2, epochs=200, batch_size=128, crf=False):
    best_loss=float("inf")
    best_loss1 = float("inf")
    a=0.5
    valid_history=[]
    train_history=[]
    for epoch in range(epochs):
        print('epoch',epoch,':', end='')
        print('epoch', epoch, ':', end='',file=of)
        for batch in batch_generator(train_X, train_y, train_y_opi, batch_size, crf=crf):
            #print(batch,' ', end='')
            batch_train_X, batch_train_y, batch_train_y_opi, batch_train_X_len, batch_train_X_mask=batch
            loss1,loss2=model(batch_train_X, batch_train_X_len, batch_train_X_mask, batch_train_y, batch_train_y_opi)
            loss=loss1+a*loss2

            optimizer2.zero_grad()
            loss2.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm(parameters2, 1.)
            optimizer2.step()


            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(parameters, 1.)
            
            optimizer.step()

        loss1,loss2=valid_loss(model, train_X, train_y, train_y_opi, crf=crf)
        loss=loss1+a*loss2
        print("train_loss: loss1: %f + loss2: %f = loss: %f" % (loss1,loss2,loss), end='')
        print("train_loss: loss1: %f + loss2: %f = loss: %f" % (loss1, loss2, loss), end='',file=of)
        train_history.append(loss)
        loss1, loss2 =valid_loss(model, valid_X, valid_y, valid_y_opi, crf=crf)
        loss = loss1 + a*loss2
        print(" || valid_loss: loss1: %f + loss2: %f = loss: %f" % (loss1,loss2,loss))
        print(" || valid_loss: loss1: %f + loss2: %f = loss: %f" % (loss1, loss2, loss),file=of)
        valid_history.append(loss)        
        if loss<best_loss and loss1<best_loss1:
            best_loss=loss
            best_loss1 = loss1
            torch.save(model, model_fn)
            print("update")
            print("update",file=of)
        shuffle_idx=np.random.permutation(len(train_X) )
        train_X=train_X[shuffle_idx]
        train_y=train_y[shuffle_idx]
        train_y_opi = train_y_opi[shuffle_idx]
    model=torch.load(model_fn) 
    return train_history, valid_history

def run(domain, data_dir, model_dir, valid_split, runs, epochs, lr, dropout, batch_size=128):
    gen_emb=np.load(data_dir+"gen.vec.npy")
    domain_emb=np.load(data_dir+domain+"_emb.vec.npy")

    ae_data=np.load(data_dir+domain+"_data.npz")

    #print("gen_emb:", gen_emb.shape)
    #print("domain_emb:", domain_emb.shape)



    valid_X=ae_data['sentences'][-valid_split:]
    valid_y=ae_data['aspect_tags'][-valid_split:]
    valid_y_opi = ae_data['opinion_tags'][-valid_split:]
    train_X=ae_data['sentences'][:-valid_split]
    train_y=ae_data['aspect_tags'][:-valid_split]
    train_y_opi = ae_data['opinion_tags'][:-valid_split]
    '''
    ae_data = np.load(data_dir + domain + ".npz")

    valid_X = ae_data['train_X'][-valid_split:]
    valid_y = ae_data['train_y'][-valid_split:]
    valid_y_opi = ae_data['train_y'][-valid_split:]
    train_X = ae_data['train_X'][:-valid_split]
    train_y = ae_data['train_y'][:-valid_split]
    train_y_opi = ae_data['train_y'][:-valid_split]
    
    print("valid_X:", valid_X.shape)
    print("valid_y:", valid_y.shape)
    print("train_X:", train_X.shape)
    print("train_y:", train_y.shape)

    print("x:")
    for i in range(0,5):
        print(train_X[i])
    print("y:")
    for i in range(0, 5):
        print(train_y[i])
    '''
    for r in range(runs):
        print(r)
        of.write(str(r))
        model=Model(gen_emb, domain_emb, 3, dropout=dropout, crf=False)
        model.cuda()
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.Adam(parameters, lr=lr)

        parameters2 = [p for p in model.parameters() if p.requires_grad]
        optimizer2 = torch.optim.Adam(parameters2, lr=lr)
        train_history, valid_history=train(train_X, train_y, train_y_opi, valid_X, valid_y, valid_y_opi, model, model_dir+domain+str(r), optimizer, parameters, optimizer2, parameters2,epochs, crf=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default="model/")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--domain', type=str, default="laptop")
    parser.add_argument('--data_dir', type=str, default="data/prep_data/")
    parser.add_argument('--valid', type=int, default=150) #number of validation data.
    parser.add_argument('--lr', type=float, default=0.0001) 
    parser.add_argument('--dropout', type=float, default=0.55) 

    args = parser.parse_args()
    of = open('out.txt', 'w')
    run(args.domain, args.data_dir, args.model_dir, args.valid, args.runs, args.epochs, args.lr, args.dropout, args.batch_size)
    of.close()
