
import tensorflow as tf
import config as cfg
import utils as pre
from sklearn import metrics
from reader import base as base_reader
import logging
import os
import time


cfg = cfg.FLAGS
data_path = cfg.data_path
out_path = cfg.out_path
if not os.path.exists(out_path):
    os.makedirs(out_path)


from importlib import import_module
Model = import_module(cfg.model_type).Model
#if model_type=="cnn":
#    from  Model_cnn import Model
#else:
#    from  Model_lstm import Model


emd_path = os.path.join(data_path,cfg.embedding_file)
emd_word_path = os.path.join(data_path,cfg.embedding_vocab)

if cfg.log_file is None:
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')
else:
    if not os.path.exists(cfg.log_path):
        os.makedirs(cfg.log_path)
    logging.basicConfig(filename=os.path.join(cfg.log_path,cfg.log_file),
                        filemode='a', level=logging.DEBUG,
                        format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')

def main():
    train_path = os.path.join(data_path,cfg.train_file)
    test_path = os.path.join(data_path,cfg.test_file)
    data_train = pre.load_data(train_path)
    data_test = pre.load_data(test_path)

    word_dict,length_voc = pre.build_voc(data_train[0]+data_test[0])

    emd_vec_path = os.path.join(data_path,cfg.embedding_file)
    emd_word_path = os.path.join(data_path,cfg.embedding_vocab)
    emd_w2v_path = os.path.join(data_path,cfg.embedding_w2v_file)

    #根据公开的senna词向量建立词向量表
    #embedding_matrix,vec_dim= pre.load_embedding(emd_vec_path,emd_word_path,word_dict)

    #根据wiki数据训练的word2vec词向量建立词向量表
    embedding_matrix,vec_dim= pre.load_w2v_embedding(emd_w2v_path,word_dict)


    max_length = max(len(max(data_train[0],key=lambda x:len(x))),len(max(data_test[0],key=lambda x:len(x))))


    cfg.length_voc = length_voc
    cfg.max_length = max_length
    cfg.sentence_vec_dim = vec_dim
    #cfg.embedding = embeddings

    train_vec = pre.dataset2id(data_train,word_dict,max_length)
    test_vec = pre.dataset2id(data_test, word_dict,max_length)

    train_batch_manager = pre.Manager_batch(train_vec,cfg.batch_size)
    test_batch_manager = pre.Manager_batch(test_vec, cfg.batch_size)

    with tf.Graph().as_default():
        with tf.name_scope("Train"):
            with tf.variable_scope("Model",reuse=False):
                train_model = Model(cfg, embedding_matrix, is_Training=True)
        with tf.name_scope("Test"):
            with tf.variable_scope("Model",reuse=True):
                valid_model = Model(cfg, embedding_matrix, is_Training=False)
            with tf.variable_scope("Model",reuse=True):
                test_model = Model(cfg, embedding_matrix, is_Training=False)
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True

        save = tf.train.Supervisor(logdir=cfg.save_path,global_step=train_model.global_steps)

        verbose = False
        with save.managed_session(config=tf_config) as sess :
            logging.info("training.....")
            best_score = 0
            best_f1 = 0
            if cfg.train:
                for epoch in range(cfg.num_epoches):
                    train_iter = train_batch_manager.iter_batch(shuffle=True)
                    test_iter = test_batch_manager.iter_batch(shuffle=False)

                    run_epoch(sess,train_model ,train_iter,is_training=True,verbose=verbose)
                    test_acc,f1= run_epoch(sess, valid_model,test_iter, is_training=False,verbose=verbose)

                    if test_acc>best_score:
                        best_score=test_acc
                        best_f1 = f1
                        if cfg.save_path:
                            save.saver.save(sess, cfg.save_path, global_step=save.global_step)
                    #logging.info('')
                    logging.info("Epoch: %d   Test: accuracy %.2f%% " %
                                 (epoch + 1,test_acc * 100))
                    print("f1:",f1)
                    #logging.info("\033[0m")
                logging.info("The best accuracy score is %.2f%%" %(best_score * 100))
                print("best f1: ",best_f1)
            if cfg.test:
                ckpt = tf.train.get_checkpoint_state(cfg.save_path)
                save.saver.restore(sess, ckpt.model_checkpoint_path)
                test_iter = test_batch_manager.iter_batch(shuffle=False)
                test_acc = evaluation(sess, test_model, test_iter)

                print('accuracy: %.2f%%' % (test_acc* 100))


def evaluation(sess, test_model, data):
    acc_count = 0
    step = 0
    predict = []
    #<##>=================
    target = []
    #</##>================
    for batch in data:
        step = step + 1
        acc, pre,lable = test_model.run_iter(sess, batch, Training=False)
        predict.extend(pre)
        #<##>=================
        target.extend(lable)
        #</##>=================
        acc_count += acc
    #print(predict)
    relations_file_path = os.path.join(data_path,cfg.relations_file)
    results_file_path = os.path.join(out_path,cfg.results_file)
    base_reader.write_results(predict, relations_file_path, results_file_path)
    #<##>=================
    base_reader.write_results(target, relations_file_path, 'out/test_keys.txt')
    #</##>=================
    #base_reader.write_results(predict, cfg.relations_file, cfg.results_file)
    return acc_count / (step * cfg.batch_size)


def run_epoch(sess, model, data_iter, is_training=True,verbose=False):
    acc_count = 0
    step = 0
    loss_count=0
    pres = []
    targets = []
    for batch in data_iter:
        step = step + 1
        start_time = time.time()
        if is_training:
            acc,loss, global_steps,lr,predict,lable= model.run_iter(sess, batch, Training=True)
            #print(A)
            acc_count += acc
            loss_count += loss
            if step % 10 == 0 and verbose == False:
                logging.info("  step: %d acc: %.2f%% loss: %.2f time: %.2f learning rate:%.5f" % (
                    step,
                    acc_count / (step * cfg.batch_size) * 100,
                    loss,
                    time.time() - start_time,
                    lr
                ))

        else:
            acc,predict,lable= model.run_iter(sess, batch,Training=False)
            acc_count += acc
        pres.extend(predict)
        targets.extend(lable)
    if is_training:
        logging.info("loss: %.2f",loss_count/step)

    #pres 样本的预测类型,如:[1, 2, 0, 7, ...]
    #target 样本的实际类型,如：[1, 5, 0, 8, ...]
    f1 = metrics.f1_score(targets, pres, average='macro')
    return acc_count / (step * cfg.batch_size),f1


if __name__ == '__main__':
    main()
