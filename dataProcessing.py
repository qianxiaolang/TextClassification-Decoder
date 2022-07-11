import pickle

import jieba

class DataProcess(object):
    def __init__(self,path,token_to_id=None,id_to_token=None):
        if token_to_id!=None:
            self.token_to_id=token_to_id
            self.id_to_token=id_to_token
        else:
            dict={}
            with open(path,encoding='utf-8') as f:
                for line in f:
                    line=line.strip()[0:-1]
                    line=line.strip()
                    for ele in line:
                        if ele not in dict:
                            dict[ele]=1
                        else:
                            dict[ele]+=1

            self.token_to_id={'<PAD>':0,'<S>':1,'<UNK>':2,'</S>':3}
            items=sorted(dict.items(),key=lambda x:-x[1])
            for i,(v,k) in enumerate(items):
                self.token_to_id[v]=i+4
            self.id_to_token={v:k for k,v in self.token_to_id.items()}

            with open('token_to_id.pkl','wb') as f:
                pickle.dump(self.token_to_id,f)

            with open('id_to_token.pkl','wb') as f:
                pickle.dump(self.id_to_token,f)

    def encode(self,path):
        sents=[]
        label=[]
        with open(path,encoding='utf-8') as f:
            for line in f:
                sent=[]
                line=line.strip()
                label.append(int(line[-1]))
                line=line[0:-1].strip()

                for ele in line:
                    if ele in self.token_to_id:
                        sent.append(self.token_to_id[ele])
                    else:
                        sent.append(self.token_to_id['<UNK>'])
                sents.append(sent)


        return sents,label





if __name__=='__main__':
    a=DataProcess('data/train.txt')
    token_to_id = {'<PAD>': 0, '<S>': 1, '<UNK>': 2, '</S>': 3}
    print(token_to_id.items())