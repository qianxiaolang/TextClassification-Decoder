import os
import pickle

import tensorflow as tf
from dataProcessing import DataProcess
from loader import DataManager
from modules import Transformer
import datetime

if os.path.isfile('token_to_id.pkl'):
    with open('token_to_id.pkl','rb') as f:
        token_to_id=pickle.load(f)
    with open('id_to_token.pkl','rb') as f:
        id_to_token=pickle.load(f)
    dataPro=DataProcess(os.path.abspath('data/train.txt'),token_to_id,id_to_token)
else:
    dataPro=DataProcess(os.path.abspath('data/train.txt'))

sents,label=dataPro.encode(os.path.abspath('data/train.txt'))

train_manager=DataManager(sents,label,batch_size=256)



inp=tf.placeholder(dtype=tf.int32,shape=[None,None])
seqLen=tf.placeholder(dtype=tf.int32,shape=[None])
tgt=tf.placeholder(dtype=tf.int32,shape=[None,None])
label=tf.placeholder(dtype=tf.int32,shape=[None])

# 下面开始构建主干网络
t=Transformer(len(dataPro.id_to_token),512)
dec=t.decoder(inp,seqLen)

left_index=tf.reshape(tf.range(tf.shape(seqLen)[0],dtype=tf.int32),shape=[-1,1])
right_index=tf.reshape(seqLen-1,shape=[-1,1])

index=tf.concat([left_index,right_index],axis=-1)
trans_output=tf.gather_nd(dec,index)

#先进行简单的全连接
ffc_output=tf.layers.dense(trans_output,units=256,activation=tf.nn.leaky_relu,kernel_initializer=tf.random_normal_initializer(stddev=0.1))
ffc_output=tf.layers.dense(ffc_output,units=128,activation=tf.nn.leaky_relu,kernel_initializer=tf.random_normal_initializer(stddev=0.1))
logits=tf.layers.dense(ffc_output,units=10,kernel_initializer=tf.random_normal_initializer(stddev=0.1))

# 下面我们是LS PLS大规模分段线性模型

Q=tf.layers.dense(ffc_output,units=60,kernel_initializer=tf.random_normal_initializer(stddev=1))
q=tf.nn.softmax(tf.reshape(Q,[-1,10,6]),axis=-1)


K=tf.layers.dense(ffc_output,units=60,kernel_initializer=tf.random_normal_initializer(stddev=1))
k=tf.reshape(K,[-1,10,6])

# logits=tf.einsum('ijm,ijm->ij',q,k)

# 这种方法可以加快收敛

loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label,logits=logits)
loss=tf.reduce_mean(loss)

op=tf.train.AdamOptimizer(learning_rate=0.0005).minimize(loss)

#开始评估准确率
model_label=tf.argmax(logits,axis=-1,output_type=tf.int32)
acc=tf.reduce_mean(tf.cast(tf.equal(model_label,label),dtype=tf.float32))

now=datetime.datetime.now()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    while True:
        for i in range(train_manager.len_data):
            inp_data, tgt_data, label_data, seqLen_data=train_manager.iter_batch(i)
            model_label_eval,acc_eval,_,loss_eval,logits_eval=sess.run([model_label,acc,op,loss,logits],feed_dict={inp:inp_data,tgt:tgt_data,label:label_data,seqLen:seqLen_data})
            print("当前模型的准确率为：{},当前的损失值为：{}".format(acc_eval,loss_eval))
            print("model_label: ",model_label_eval)
            print("label: ",label_data)
            if i%10==0:
                print("当前时间为：",datetime.datetime.now()-now)


# 六分钟，adm=0.0005，可以达到百分之97
