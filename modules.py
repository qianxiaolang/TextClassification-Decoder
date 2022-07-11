import tensorflow as tf

class Transformer(object):
    def __init__(self,vocab_size,d_model):
        self.test_dict={}
        self.d_model=d_model
        self.embedding=tf.get_variable(name='embedding'
                                       ,shape=[vocab_size,d_model]
                                       ,initializer=tf.random_normal_initializer(stddev=0.1)
                                       ,dtype=tf.float32)
        self.pos_embedding=self.get_position_embedding(128,d_model)

    def get_position_embedding(self,word_num=128,d_model=128):
        inv = 1 / (10000 ** (tf.range(0, d_model, 2, dtype=tf.float32) / d_model))
        n = tf.range(word_num, dtype=tf.float32)
        pos = tf.einsum('i,j->ij', n, inv)
        pos = tf.concat([tf.sin(pos), tf.cos(pos)], axis=-1)
        return pos

    def scaled_dot_product_attention(self,q,k,v,mask,dropout_rate=0.1):
        with tf.variable_scope('scaled_dot_product_attention',reuse=tf.AUTO_REUSE):
            outputs=tf.einsum('ijm,ikm->ijk',q,k)
            outputs=outputs/(q.get_shape().as_list()[-1]**0.5)
            #下面进行序列化的mask
            mask=tf.tile(mask,[tf.shape(q)[0]//tf.shape(mask)[0],1])
            mask=tf.expand_dims(mask,axis=1)
            outputs+=(mask*(-2**20))
            #下面进行因果序列化的mask
            ones=tf.ones([tf.shape(q)[1],tf.shape(q)[1]])
            ones=1-tf.matrix_band_part(ones,-1,0)
            outputs+=(ones*(-2**20))

            outputs=tf.nn.softmax(outputs)
            self.test_dict['weight']=outputs

            outputs=tf.einsum('ijk,ikm->ijm',outputs,v)
            return outputs

    def ln(self,inputs, epsilon=1e-8, scope='ln'):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]

            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
            gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
            normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
            outputs = gamma * normalized + beta

        return outputs

    def multihead_attention(self,queries,keys,values,mask,num_head=8,d_model=64,dropout_rate=0.1):
        with tf.variable_scope('multihead_attention',reuse=tf.AUTO_REUSE):
            Q=tf.layers.dense(queries,num_head*d_model,use_bias=True,kernel_initializer=tf.random_normal_initializer(stddev=0.1))
            K=tf.layers.dense(keys,num_head*d_model,use_bias=True,kernel_initializer=tf.random_normal_initializer(stddev=0.1))
            V=tf.layers.dense(values,num_head*d_model,use_bias=True,kernel_initializer=tf.random_normal_initializer(stddev=0.1))

            q=tf.concat(tf.split(Q,num_head,axis=-1),0)
            k=tf.concat(tf.split(K,num_head,axis=-1),0)
            v=tf.concat(tf.split(V,num_head,axis=-1),0)

            outputs=self.scaled_dot_product_attention(q,k,v,mask)

            outputs=tf.concat(tf.split(outputs,axis=0,num_or_size_splits=num_head),axis=-1)

            outputs=tf.layers.dense(outputs,units=self.d_model)

            outputs+=queries

            outputs=self.ln(outputs)

            return outputs

    def ffc(self,input,num_units=512):
        with tf.variable_scope('ffc',reuse=tf.AUTO_REUSE):
            outputs=tf.layers.dense(input,num_units,activation=tf.nn.leaky_relu)
            outputs=tf.layers.dense(outputs,self.d_model)

            outputs+=input

            outputs=self.ln(outputs)

            return outputs

    def decoder(self,inp,seq_len,num_blocks=1):
        inp_embedding=tf.nn.embedding_lookup(self.embedding,inp)
        dec=inp_embedding+self.pos_embedding[0:tf.shape(inp_embedding)[1]]
        mask = 1 - tf.to_float(tf.sequence_mask(seq_len))
        with tf.variable_scope('decoder',reuse=tf.AUTO_REUSE):
            for i in range(num_blocks):
                with tf.variable_scope('num_blocks_{}'.format(i),reuse=tf.AUTO_REUSE):
                    dec=self.multihead_attention(dec,dec,dec,mask)

                    dec=self.ffc(dec,2*self.d_model)

            return dec



