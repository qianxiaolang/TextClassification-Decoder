import numpy as np


class DataManager(object):
    def __init__(self,sents,label,batch_size=64):
        self.label=[]
        self.inp_data=[]
        self.tgt_data=[]
        self.length=[]
        self.sort_and_pad(sents,label,batch_size)

    def sort_and_pad(self,sents,label,batch_size):
        #'<PAD>':0,'<S>':1,'<UNK>':2,'</S>':3
        self.len_data=int(np.ceil(len(sents)/batch_size))
        for i in range(self.len_data):
            sub_sents=sents[i*batch_size:(i+1)*batch_size]
            inp=[]
            tgt=[]
            sub_len=[]
            sub_label=label[i*batch_size:(i+1)*batch_size]
            max_len=max([len(ele) for ele in sub_sents])
            for ele in sub_sents:
                sub_len.append(len(ele)+1)
                tgt.append(ele+[3]+(max_len-len(ele))*[0])
                inp.append([1]+ele+(max_len-len(ele))*[0])

            self.label.append(sub_label)
            self.inp_data.append(inp)
            self.tgt_data.append(tgt)
            self.length.append(sub_len)

    def iter_batch(self,index=0):

        return self.inp_data[index],self.tgt_data[index],self.label[index],self.length[index]


