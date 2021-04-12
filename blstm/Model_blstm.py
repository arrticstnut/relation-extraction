import tensorflow as tf
import numpy as np
from sklearn import metrics
from tensorflow.contrib import rnn
from tensorflow.contrib.layers.python.layers import initializers

class Model(object):
    #def __init__(self,config,embedding_matrix=None,is_Training=True):
    #def __init__(self,config,is_Training=True):
    def __init__(self,config,embedding_matrix,is_Training=True):
        self.config = config
        self.is_Training = is_Training
        self.batch_size = config.batch_size
        self.max_len = config.max_length
        self.r_class = config.class_num  # number of relations
        self.num_filters = config.num_filters
        self.input_feature = config.sentence_vec_dim+self.config.pos_embedding_size*2
        self.dropout = config.keep_prob
        self.h_filters_windows=config.h_filters_windows
        self.margin = config.margin

        #正则化系数
        self.regular_lambda = config.regular_lambda

        self.lr = config.learning_rate
        self.decay = config.learning_decay
        self.loss_l2 = 0

        self.init_value = tf.truncated_normal_initializer(stddev=0.1)
        self.initializer_layer = initializers.xavier_initializer()

        with tf.name_scope("input_layer"):
            #
            #输入
            #
            #   self.input_sentences 单个批次单词index向量 = shape[batch_size, max_len]
            #   self.distant_1 = 单个批次单词的位置向量 = shape[batch_size, max_len]
            #   初值是随机是在训练的时候由feed_dict喂的，见Model_cnn.py:run_iter）
            #   self.distant_2 类同
            #   self.input_relation = 单个批次的句子实体对的关系类型向量= shape[batch_size, 1]
            #   self.input = (self.input_sentences,self.distant_1,self.distant_2,self.input_relation)

            self.input_sentences = tf.placeholder(dtype=tf.int32,shape=[self.batch_size,self.max_len],
                                                  name="input_S")

            self.distant_1 = tf.placeholder(dtype=tf.int32,shape=[self.batch_size,self.max_len],
                                            name="dist_e1")
            self.distant_2 = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.max_len],
                                            name="dist_e1")
            self.input_relation = tf.placeholder(dtype=tf.int32,shape=[self.batch_size],name="lable")

            self.input = (self.input_sentences,self.distant_1,self.distant_2,self.input_relation)


        with tf.name_scope("embedding_matrix"):
            init_value = tf.truncated_normal_initializer(stddev=0.1)
            if embedding_matrix is None:
                self.embedding_tab = tf.get_variable(name="sent_embedding_matrix",dtype=tf.float32,
                                                   shape=[self.config.length_voc,self.config.sentence_vec_dim])
            else:
                self.embedding_tab = tf.get_variable(name="sent_embedding_matrix",dtype=tf.float32,
                                                   initializer=embedding_matrix)#sentence,e1,e2
                #self.dist1_tab = tf.get_variable(name="pos1_embedding_matrix",dtype=tf.float32,
                #                                 shape=[self.max_len,self.config.pos_embedding_size])
                #self.dist2_tab = tf.get_variable(name="pos2_embedding_matrix",dtype=tf.float32,
                #                                 shape=[self.max_len,self.config.pos_embedding_size])
                #self.dist1_tab = tf.get_variable(name="pos1_embedding_matrix",dtype=tf.float32,
                #                                 shape=[2*self.max_len,self.config.pos_embedding_size])
                #self.dist2_tab = tf.get_variable(name="pos2_embedding_matrix",dtype=tf.float32,
                #                                 shape=[2*self.max_len,self.config.pos_embedding_size])
                #self.r_class = config.class_num
                #self.input_relation -> [batch_size, 1]
                #self.labels --> [batch_size, class_num]
                #   每一行为该句子的关系类别的one-hot向量
                self.labels = tf.one_hot(self.input_relation, self.r_class)
        
        with tf.name_scope("forward"):

            #创建样本的输入embedding向量
            input_feature= self.embedding_layer()
            # --> [batch_size, max_len, dim]
            if is_Training:
                input_feature = tf.nn.dropout(input_feature, self.dropout)

            #通过lstm层
            lstm_out = self.lstm_layer(input_feature)
            # --> [batch_size , max_len , 2*self.num_filters]
            #if is_Training:
                #lstm_out = tf.nn.dropout(lstm_out, self.dropout)
            
            '''
            #通过attention层
            #
            #feature = self.attention_layer(lstm_out)
            # --> [batch_size, 2*num_filter]
            #
            feature_size = feature.shape.as_list()[1]
            '''
            feature = tf.reshape(lstm_out,[self.batch_size, -1])
            # --> [batch_size, n*num_filter]
            feature_size = feature.shape.as_list()[1]

            #全连接层
            # --> [batch_size, num_class]
            logits,_ = self.predict_layer(feature,feature_size,self.r_class)

            #预测
            #self.predict = 单批次中每个logits的最大值所在的下标 --> [batch_size]
            self.predict = tf.argmax(logits,axis=1,output_type=tf.int64) #batch_size

            #accuracy = self.predict 和 self.labels的值对应的下标是否一致的出来的向量。
            #   -->[batch_size]
            #   值为[1,0,1,0 ,...]等组成的向量
            accuracy = tf.equal(self.predict, tf.argmax(self.labels, axis=1))

            #self.acc = accuracy中1的个数，即预测正确的样本的个数。是标量
            self.acc = tf.reduce_sum(tf.cast(accuracy, tf.float32))

        if is_Training:
            with tf.name_scope("loss"):
                #<##>=========================================
                #self.loss_l2 = tf.contrib.layers.apply_regularization(
                #    regularizer=tf.contrib.layers.l2_regularizer(self.regular_lambda),
                #    weights_list=tf.trainable_variables())
                #</##>=========================================
                self.loss_ce = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=logits))
                self.loss = self.loss_ce + self.loss_l2

            #优化器
            with tf.name_scope("optimizer"):
                #全局步数
                global_steps = tf.get_variable(name="global_steps",initializer=0,trainable=False)
                #学习率
                self.lr = tf.train.exponential_decay(self.lr, global_steps, self.batch_size*2, self.decay, staircase=True)
                #优化器
                opt = tf.train.AdamOptimizer(self.lr)
                #最小化
                self.train_op = opt.minimize(loss=self.loss,global_step=global_steps)
                #边界操作（无实际运算）
                self.reg_op = tf.no_op()
                self.global_steps = global_steps

    def embedding_layer(self):
            #
            #查找embedding向量
            #
            with tf.name_scope("embedding_layer"):
                sent_emb = tf.nn.embedding_lookup(self.embedding_tab,self.input_sentences,
                                                  name="sents_embedding")#bz,n,dim-sent
                #dist1_emb = tf.nn.embedding_lookup(self.dist1_tab,self.distant_1,
                #                                   name="position1_embedding")#bz,n,dim-pos
                #dist2_emb = tf.nn.embedding_lookup(self.dist2_tab,self.distant_2,
                #                                   name="position2_embedding")#bz,n,dim-pos
                #input_feature = tf.concat([sent_emb, dist1_emb, dist2_emb], axis=-1, name="input_feature")#bz,n,dim-sent+pos1+pos2
                input_feature = tf.concat([sent_emb], axis=-1, name="input_feature")#bz,n,dim-sent
                #input_feature = tf.reshape(input_feature,
                #                           [self.batch_size,self.max_len,self.input_feature,1])#bz,n,dim-sent+pos1+pos2,1

            return input_feature

    def lstm_layer(self,input_feature):
            #
            #BLSTM层
            #input_feature：输入特征，维度[batch_size,max_len,dim_sen + dim_pos1 + dim_pos2]
            #
            with tf.variable_scope("Bi-lstm_layer"):
                fw_cell = rnn.LSTMCell(self.num_filters, use_peepholes=True,
                                       initializer=self.initializer_layer, state_is_tuple=True)
                bw_cell = rnn.LSTMCell(self.num_filters, use_peepholes=True,
                                       initializer=self.initializer_layer, state_is_tuple=True)
                output, output_state = tf.nn.bidirectional_dynamic_rnn(cell_bw=bw_cell,
                                                                       cell_fw=fw_cell,
                                                                       dtype=tf.float32,
                                                                       inputs=input_feature)
                #outout为BLSTM的输出， 是一个元组，为（前向output，后向output)
                #out_lstm将前后想的BLSTM的输出拼接在一起
                out_lstm = tf.concat(output, axis=2) #[batch_size , max_len , 2*self.num_filters]

                #只需获取output最后一步即可
                out_lstm = out_lstm[:,-1,:]
                # -> [batch_size, 1, n*num_filters]
                #<TEST>============================
                print(out_lstm.shape.as_list())
                #</TEST>============================

                return out_lstm

    '''
    def attention_layer(self,lstm_feature):
            #
            #attencion层
            #lstm_feature:维度,[batch_size, max_len, 2*num_filter]
            #attention求解过程的理解：
            #   不考虑batch_size：
            #       lstm的输出lstm_o的维度：[max_len, 2*num_filter]
            #       最后注意力alph的维度: [1, max_len] 
            #       注意力作用在lstm上后，去掉了max_len维度: alph x lstm_o --> [1, 2*num_filter]
            #   再在考虑batch_size的情况下进行上述操作过程

            with tf.name_scope("attention_layer"):

                ##(1).求attention的权重alph
                #W,[2*num_filters, 1]
                attention_w = tf.get_variable('attention_omega', [2*self.num_filters, 1],trainable=True)

                #tanh --> [batch_size, max_len, 2*num_filter]
                tanh_lstm_feature = tf.tanh(lstm_feature)

                #reshape --> [batch_size*max_len, 2*num_filter]
                tanh_lstm_feature = tf.reshape(tanh_lstm_feature, [self.batch_size * self.max_len, -1])

                #1)tanh_lstm_feature x attention_w -->[batch_size*max_len, 2*num_filter] x [2*num_filters, 1] 
                #                               --> [batch_size*max_len, 1]
                #2)reashape --> [batch_size, max_len]
                #3)sotfmax --> [batch_size, max_len]
                #4)reshape --> [batch_size, 1, max_len]
                alph = tf.nn.softmax(tf.reshape(tf.matmul(tanh_lstm_feature,attention_w),[self.batch_size,self.max_len]))
                alph = tf.reshape(alph,[self.batch_size,1,self.max_len])

                ##(2).将得到的alph作用在lstm的输出上
                #1)alph x lstm_feature --> [batch_size, 1, max_len] x [batch_size, max_len, 2*num_filter]
                #                       -->[batch_size, 1, 2*num_filter]
                #2)reshape --> [batch_size, 2*num_filter]
                #3)tanh --> [batch_size, 2*num_filter]
                out_att = tf.nn.tanh(tf.reshape(tf.matmul(alph,lstm_feature),[self.batch_size,2*self.num_filters]))#bz,num_filter

                #<#>-------------------------------------------
                self.loss_l2 += tf.contrib.layers.l2_regularizer(self.regular_lambda)(attention_w)
                #</#>-------------------------------------------

            return out_att

            '''

    def predict_layer(self,feature,feature_size,num_class):
        #
        #feature:维度[batch_size, n*num_filters]
        #feature_size: n*num_filters
        #其中n为:
        #   (cnn,不同高度卷积核的数量，如3、4、5数量为3)
        #   (blstm,2)
        #   (lstm,1)
        in_size = feature_size
        out_size = num_class
        with tf.name_scope("predict_liner_layer"):
            loss_l2 = tf.constant(0, dtype=tf.float32)
            w = tf.get_variable('linear_W', [in_size, out_size],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
            b = tf.get_variable('linear_b', [out_size],
                                initializer=tf.constant_initializer(0.1))

            # --> [batch_size, out_size]
            o = tf.nn.xw_plus_b(feature, w, b)

            #loss_l2 += tf.nn.l2_loss(w) + tf.nn.l2_loss(b)
            #loss_l2 += tf.nn.l2_loss(w)
            self.loss_l2 += tf.contrib.layers.l2_regularizer(self.regular_lambda)(w)

            #<##>------------------------
            #将l2_loss加入loss集合
            #tf.add_to_collection("loss_regular",tf.nn.l2_loss(w))
            #</##>------------------------
            return o, loss_l2

    def creat_feed(self,Training,batch):
        #
        #创建feed_dict
        #

        #batch_zip = (第i批单词索引向量，e1索引值，e2索引值，各单词到e1的距离向量，到e2的距离向量，关系类别)
        #sentences_id = 单词索引向量 = 维度 [batch_size , max_len ]
        #e1_vec = e1索引值 = 维度[batch_size , 1 ]
        #e2_vec = e2索引值 = 维度[batch_size , 1 ] 
        #dist_e1 = 各单词到e1的距离向量 = 维度 [batch_size , max_len ]
        #dist_e2 = 到e2的距离向量 = 维度 [batch_size , max_len ]
        #relation  = 句子的实际关系类别 = 维度[batch_size , 1 ] 
        batch_zip = (x for x in zip(*batch))
        sentences_id, e1_vec, e2_vec, dist_e1, dist_e2, relation= batch_zip
        in_sents,in_dist1,in_dist2,rel=self.input
        feed_dict = {in_sents: np.asarray(sentences_id),in_dist1:dist_e1,
                     in_dist2:dist_e2,rel:relation}
        if Training:
            feed_dict[rel] = relation
        return feed_dict

    def run_iter(self,sess, batch, Training=False):
        feed_dict = self.creat_feed(Training,batch)
        if Training:
            _,_, acc,lr, loss,step ,predict,lable,= sess.run([self.train_op, self.reg_op,self.acc,self.lr,
                                                    self.loss,self.global_steps,self.predict,self.input_relation], feed_dict=feed_dict)

            #predict 预测的类型 -> [batch_size]
            #lable 实际的类型 -> [batch_size]
            predict = list(predict)
            lable = list(lable)

            #f1 = metrics.f1_score(lable,predict,average='macro')
            return acc,loss,step,lr,predict,lable
        else:
            acc,predict,lable= sess.run([self.acc,self.predict,self.input_relation], feed_dict=feed_dict)
            predict = list(predict)
            lable = list(lable)
            #f1 = metrics.f1_score(lable, predict, average='macro')
            return acc,predict,lable

