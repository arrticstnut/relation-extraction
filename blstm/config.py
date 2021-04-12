import tensorflow as tf

tf.app.flags.DEFINE_string("model_type", "Model_blstm", "model file")
tf.app.flags.DEFINE_float("regular_lambda",0.0001,"ragular lambda value")

tf.app.flags.DEFINE_string("data_path","../data/","data root path")

tf.app.flags.DEFINE_string("train_file", "SemEval_2010_task_8/train.txt", "training file")
tf.app.flags.DEFINE_string("test_file", "SemEval_2010_task_8/test.txt", "Test file")

tf.app.flags.DEFINE_integer("batch_size",200,"batch size")

#tf.app.flags.DEFINE_integer("pos_embedding_size",5,"position embedding size")
tf.app.flags.DEFINE_integer("pos_embedding_size",0,"position embedding size")

tf.app.flags.DEFINE_integer("num_filters",500,
                            "How many features a convolution op have to output")
tf.app.flags.DEFINE_integer("class_num", 19, "Number of relations")
tf.app.flags.DEFINE_float("keep_prob", 0.5, "Dropout keep prob.")

tf.app.flags.DEFINE_integer("h_filters_windows",3,"the height of filters window ")
tf.app.flags.DEFINE_float("margin", 1, "margin based loss function")
tf.app.flags.DEFINE_float("learning_rate", 0.003, "the learning rate of trainning")
tf.app.flags.DEFINE_float("learning_decay", 0.98, "the learning decay rate of trainning")
tf.app.flags.DEFINE_integer("num_epoches", 200, "Number of epoches")

tf.app.flags.DEFINE_string("log_file", "logfile.log", "Log file")
tf.app.flags.DEFINE_string("save_path","./model/", "save model here")
tf.app.flags.DEFINE_bool('train', True, 'set True to Train')
tf.app.flags.DEFINE_bool('test', False, 'set True to test')


#------------------------------------------------
tf.app.flags.DEFINE_string("relations_file", "SemEval_2010_task_8/relations.txt", "relations file")#实际关系标签所在的文件
tf.app.flags.DEFINE_string("results_file", "results.txt", "predicted results file")#输出文件名

tf.app.flags.DEFINE_string("out_path", "./out", "输出文件的路径")
tf.app.flags.DEFINE_string("log_path", "./log", "日志文件的路径")
#------------------------------------------------

tf.app.flags.DEFINE_string("embedding_w2v_file","embedding/wiki_embedding_w2v/wiki_embedding_w2v_d300.vector","word2vec embedding file")#用wiki语料训练的word2vec词向量文件
tf.app.flags.DEFINE_integer("length_voc",1,"单词总量") #单词总量
tf.app.flags.DEFINE_integer("max_length",1,"句子的最大长度") #句子的最大长度
tf.app.flags.DEFINE_integer("sentence_vec_dim",1,"单词embed向量的维度") #单词embed向量的维度
tf.app.flags.DEFINE_float("embedding",0,"单词的词嵌入向量矩阵")

FLAGS=tf.app.flags.FLAGS
