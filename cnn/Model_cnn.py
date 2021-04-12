import tensorflow as tf
import numpy as np
from sklearn import metrics
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
        self.input_feature_dim = config.sentence_vec_dim+self.config.pos_embedding_size*2
        self.dropout = config.keep_prob
        self.h_filters_windows=config.h_filters_windows
        self.margin = config.margin
        self.regular_lambda = config.regular_lambda

        self.lr = config.learning_rate
        self.decay = config.learning_decay
        self.loss_l2 = 0

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
                self.dist1_tab = tf.get_variable(name="pos1_embedding_matrix",dtype=tf.float32,
                                                 shape=[2*self.max_len,self.config.pos_embedding_size])
                self.dist2_tab = tf.get_variable(name="pos2_embedding_matrix",dtype=tf.float32,
                                                 shape=[2*self.max_len,self.config.pos_embedding_size])
                #self.r_class = config.class_num
                #self.input_relation -> [batch_size, 1]
                #self.labels --> [batch_size, class_num]
                #   每一行为该句子的关系类别的one-hot向量
                self.labels = tf.one_hot(self.input_relation, self.r_class)


        with tf.name_scope("forward"):
            #embedding look-up
            #输入层
            input_feature,sent_emb= self.embedding_layer()

            #if is_Training:
            #    input_feature = tf.nn.dropout(input_feature, self.dropout)

            #卷积层输出
            # --> [batch_size, 3*num_filter]
            feature = self.convolution_layer(input_feature,init_value)
            if is_Training:
                feature = tf.nn.dropout(feature, self.dropout)
            feature_size = feature.shape.as_list()[1]

            #全连接层
            # --> [batch_size, num_class]
            logits = self.predict_layer(feature,feature_size,self.r_class)

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
            #损失函数
            with tf.name_scope("loss"):
                self.loss_ce = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=logits))
                #self.loss = self.loss_ce+0.01*self.loss_l2
                #self.loss = self.loss_ce + self.regular_lambda*tf.add_n(tf.get_collection("loss_regular"))
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
            input_feature = tf.concat([sent_emb], axis=-1, name="input_feature")#bz,n,dim-sent+pos1+pos2
            #input_feature = tf.reshape(input_feature,
            #                           [self.batch_size,self.max_len,self.input_feature_dim,1])#bz,n,dim-sent+pos1+pos2,1


        return input_feature,sent_emb


    def convolution_layer(self,input_data,initializer):
        #
        # input_data: batch_size,max_len(句子最大长度),dim_sent(词向量维度)+2*dim_pos(位置向量维度)
        #
        with tf.name_scope("convolution_layer"):
            #将输入数据的维度reshape为4️维，方便当成图像输入，最后一维的通道为1
            input_data = tf.reshape(input_data,[-1, self.max_len,self.input_feature_dim,1])
            
            #h_windows = self.h_filters_windows

            #图像的宽度
            w_windows = self.input_feature_dim

            pool_outputs = []
            window_heights = [3]
            for filter_size in window_heights:
                with tf.variable_scope('conv-%s' % filter_size):
                    #卷积核
                    cnn_w = tf.get_variable(shape=[filter_size,w_windows,1,self.num_filters],
                                            initializer=initializer,name="cnn_w")

                    #cnn_b = tf.get_variable(shape=[self.num_filters],initializer=tf.constant_initializer(0.1),name="cnn_b")
                    cnn_b = tf.get_variable(shape=[self.num_filters],initializer=tf.truncated_normal_initializer(stddev=0.1),name="cnn_b")
                    #卷积核在数据上卷积后的输出,输出维度[batch_size ,ceil(max_len /stride) , 1 , num_filters]
                    conv = tf.nn.conv2d(input_data,cnn_w,strides=[1,1,self.input_feature_dim,1],padding="SAME")

                    #加上bias的激活
                    R = tf.nn.relu(tf.nn.bias_add(conv,cnn_b),name="R") #bz,n,1,n_filters

                    #最大池化，输出维度[batch_size, 1, 1, num_filters]
                    R_pool = tf.nn.max_pool(R, ksize=[1,self.max_len,1 , 1],
                                        strides=[1,self.max_len,1, 1]
                                        , padding="SAME")
                    pool_outputs.append(R_pool)

            #将不同高度窗口得到的输出拼接在一起
            pools = tf.reshape(tf.concat(pool_outputs, 3), [-1, len(window_heights) * self.num_filters])
            #pools --> [batch_size, 3*self.num_filter]

            return pools

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
            o = tf.nn.xw_plus_b(feature, w, b)  # batch_size, out_size

            #loss_l2 += tf.nn.l2_loss(w) + tf.nn.l2_loss(b)
            self.loss_l2 += self.regular_lambda*tf.nn.l2_loss(w)
            

            #<##>------------------------
            #将l2_loss加入loss集合
            #tf.add_to_collection("loss_reg",tf.)
            #tf.add_to_collection("loss_regular",tf.nn.l2_loss(w))
            #</##>------------------------
            return o



    def creat_feed(self,Training,batch):
        #
        #创建feed_dict
        #

        #   batch_zip = (第i批单词索引向量，e1索引值，e2索引值，各单词到e1的距离向量，到e2的距离向量，关系类别)
        #   sentences_id = 单词索引向量 = 维度 [batch_size , max_len ]
        #   e1_vec = e1索引值 = 维度[batch_size , 1 ]
        #   e2_vec = e2索引值 = 维度[batch_size , 1 ] 
        #   dist_e1 = 各单词到e1的距离向量 = 维度 [batch_size , max_len ]
        #   dist_e2 = 到e2的距离向量 = 维度 [batch_size , max_len ]
        #   relation  = 句子的实际关系类别 = 维度[batch_size , 1 ] 

        batch_zip = (x for x in zip(*batch))
        sentences_id, e1_vec, e2_vec, dist_e1, dist_e2, relation= batch_zip
        in_sents,in_dist1,in_dist2,rel=self.input
        feed_dict = {in_sents: np.asarray(sentences_id),in_dist1:dist_e1,
                     in_dist2:dist_e2,rel:relation}
        if Training:
            feed_dict[rel] = relation
        return feed_dict

    """
    def run_epoch(self,sess, batch, Training=False):
        feed_dict = self.creat_feed(Training,batch)
        if Training:
            _,_, acc,lr, loss,step ,predict,lable= sess.run([self.train_op, self.reg_op,self.acc,self.lr,
                                                    self.loss,self.global_steps,self.predict,self.input_relation], feed_dict=feed_dict)
            predict = list(predict)
            lable = list(lable)
            f1 = metrics.f1_score(lable,predict,average='macro')
            return acc,f1,loss,step,lr
        else:
            acc,predict,lable= sess.run([self.acc,self.predict,self.input_relation], feed_dict=feed_dict)
            predict = list(predict)
            lable = list(lable)
            f1 = metrics.f1_score(lable, predict, average='macro')
            return acc,f1,predict

    """
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




