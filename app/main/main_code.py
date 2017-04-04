# -*- coding: utf-8 -*-
"""
主要的函数定义
@author: lvrr
"""
import os
import lda
import time
import jieba
import codecs
import pickle
import logging
import numpy as np
import configparser
from functools import wraps


from sklearn import metrics
from sklearn.svm import SVC
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

logger_info = logging.getLogger('info')
logger_error = logging.getLogger('error')
logger_time = logging.getLogger('time')


def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        logger_time.info("Total time running %s : %s seconds" %
                    (function.__name__, str(t1 - t0)))

        return result

    return function_timer

@fn_timer
def load(filename):
    # 序列化的输出
    with open(filename, 'rb') as output:
        data = pickle.load(output)
    return data


@fn_timer
def save(filename, data):
    # 序列化的输出
    with open(filename, 'wb') as output:
        pickle.dump(data, output)


def cut_jieba(sentence):
    word_list = list(jieba.cut(sentence))
    return word_list


class BaseFuction(object):
    """docstring for BaseFuction"""

    def __init__(self):
        super(BaseFuction, self).__init__()

        self.conf = os.path.join(os.path.dirname(__file__), 'initialization.conf')

        self.cf = self.read_conf()

        self.courps_path = self.cf.get('datasets_path','corpus')
        self.stopword_path = self.cf.get('datasets_path','stopword')
        self.cut_path = self.cf.get('datasets_path','cut_corpus')
        self.label_path = self.cf.get('datasets_path','label')
        self.unlabel_path = self.cf.get('datasets_path', 'unlabel')

        self.dic_path = self.cf.get('vector_path','dic')
        self.tfidf_path = self.cf.get('vector_path','tfidf')
        self.lda_path = self.cf.get('vector_path','lda')
        self.chi2_path = self.cf.get('vector_path','chi2')

        self.lda_iter = self.cf.getint('model_parameter','lda_iter')
        self.lda_dim = self.cf.getint('model_parameter','lda_dim')
        self.feature_chi2 = self.cf.getint('model_parameter','feature_chi2')

        self.classify_path = self.cf.get('classifier_path','classifier')

        self.svm_C = self.cf.getfloat('classifier_parameter','svm_C')
        self.svm_kernel = self.cf.get('classifier_parameter','svm_kernel')

        self.labels = (self.cf.get('labels_parameter','labels')).split(' ')

        self.stopword_list = self.read_stopwords()

    def read_conf(self):
        """初始化读取配置文件"""
        cf = configparser.ConfigParser()
        cf.read(self.conf, encoding='UTF-8')
        return cf

    def write_conf(self):
        """写配置文件"""
        with open(self.conf,'w') as f:
            self.cf.write(f)


    @fn_timer
    def read_stopwords(self):
        """读停用词表"""
        stopwords = list()
        input_file = codecs.open(self.stopword_path, 'r', encoding='utf-8')
        for word in input_file:
            stopwords.append(word.strip())
        input_file.close()
        if stopwords:
             return np.array(stopwords)
        else:
            logger_error.error('%s',ValueError('read stopword list fail'))
            raise ValueError('read stopword list fail')

    def cut_jieba_stop(self, sentence):
        save_list = list()
        word_list = cut_jieba(sentence)
        for word in word_list:
            if (len(word) > 1) and (word not in self.stopword_list):
                save_list.append(word)
        return save_list

    def split_data_random(self, doc_list, tag_list):
        train_set, test_set, train_tag, test_tag = train_test_split(
            doc_list, tag_list, test_size=0.8, random_state=0)

        return train_set, test_set, train_tag, test_tag


class TrainModel(BaseFuction):
    """训练向量化的模型"""

    def __init__(self):
        super(TrainModel, self).__init__()

    @staticmethod
    def LineSentence(filename):
        """将文本迭代的输入"""
        with codecs.open(filename, "r", encoding='utf-8') as text_file:
            for line in text_file:
                yield line

    @fn_timer
    def train_dic_model(self):
        """ 训练dic model"""

        dic_filename = self.dic_path
        if os.path.exists(dic_filename):
            logger_info.info('dict exists：{}'.format(dic_filename))
            dic = load(dic_filename)
        else:
            vectorizer = CountVectorizer()
            dic = vectorizer.fit(self.LineSentence(self.cut_path))
            save(dic_filename, dic)
            logger_info.info('save dict success')
        return dic

    @fn_timer
    def train_lda_model(self):
        """ 训练lda model"""

        dic = self.train_dic_model()
        tf_mat = dic.transform(self.LineSentence(self.cut_path))

        model = lda.LDA(n_topics= self.lda_dim, n_iter= self.lda_iter,
                        eta=0.01, alpha=0.1, random_state = 0)
        lda_model = model.fit(tf_mat)
        save(self.lda_path, lda_model)
        logger_info.info('train lda model success')

    @fn_timer
    def train_tfidf_model(self):
        """ 训练tfidf model"""
        dic = self.train_dic_model()
        tf_mat = dic.transform(self.LineSentence(self.cut_path))

        transformer = TfidfTransformer()
        tfidf = transformer.fit(tf_mat)
        save(self.tfidf_path, tfidf)
        logger_info.info('train tfidf model success')


class Vectorizer(BaseFuction):
    """将文本向量化表示"""

    def __init__(self):
        super(Vectorizer, self).__init__()

    @fn_timer
    def train_chi2(self, doc_list, tag_list):
        dic = load(self.dic_path)
        dic_list = dic.transform(doc_list)

        tfidf = load(self.tfidf_path)
        tfidf_list = tfidf.transform(dic_list)

        chi2_v = SelectKBest(chi2, k=self.feature_chi2).fit(tfidf_list, tag_list)
        save(self.chi2_path, chi2_v)
        logger_info.info('train chi2 model success')

    @fn_timer
    def output_tfidf_vertor(self, doc_list, tag_list=list()):
        # 利用tfidf模型将文本转换
        # 生成tfidf模型

        dic = load(self.dic_path)
        dic_list = dic.transform(doc_list)

        tfidf = load(self.tfidf_path)
        tfidf_list = tfidf.transform(dic_list)

        chi2_v = load(self.chi2_path)
        tfidf_chi2 = chi2_v.transform(tfidf_list).toarray()
        return tfidf_chi2

    @fn_timer
    def output_lda_vertor(self, doc_list):
        # 利用lda模型将文本转换
        # 生成lda模型

        dic = load(self.dic_path)
        dic_list = dic.transform(doc_list)

        lda_model = load(self.lda_path)
        lda_list = lda_model.transform(dic_list)

        return np.array(lda_list)

    @fn_timer
    def output_merge_vertor(self, doc_list):
        tfidf_chi2 = self.output_lda_vertor(doc_list)
        lda_list = self.output_tfidf_vertor(doc_list)

        merge_list = list()
        for index in range(len(doc_list)):
            merge_list.append(np.append(lda_list[index], tfidf_chi2[index]))

        return np.array(merge_list)


class Classify(BaseFuction):
    """训练分类器，预测分类结果"""

    def __init__(self):
        super(Classify, self).__init__()

    @staticmethod
    def calculate_result(data_tag, data_pred):
        # 对训练的分类器测评
        m_accuracy = metrics.accuracy_score(data_tag, data_pred)
        m_precision = metrics.precision_score(data_tag, data_pred, average=None)
        m_recall = metrics.recall_score(data_tag, data_pred, average=None)
        m_f1 = metrics.f1_score(data_tag, data_pred, average=None)
        logger_info.info('accuracy:{}'.format(m_accuracy))
        for tag_index in range(len(set(data_tag))):
            logger_info.info('tag:{},precision:{},recall:{},f1:{}'\
                .format(tag_index,m_precision[tag_index],m_recall[tag_index],m_f1[tag_index]))

        return m_accuracy

    @fn_timer
    def train_svm_classify(self, train_set, test_set, train_tag, test_tag):
        # 训练SVM分类器

        clf = SVC(C=self.svm_C, kernel=self.svm_kernel, degree=3,
                  gamma='auto', coef0=0.0, shrinking=True,
                  probability=True, tol=0.001, cache_size=200,
                  class_weight=None, verbose=False, max_iter=-1,
                  decision_function_shape='ovo',
                  random_state=None)

        clf_res = clf.fit(train_set, train_tag)
        logger_info.info('clf_rec:{}'.format(clf_res))
        save(self.classify_path, clf_res)
        logger_info.info('train classify success')

        train_pred = clf_res.predict(train_set)
        logger_info.info('train data:')
        result_train = self.calculate_result(train_tag, train_pred)

        test_pred = clf_res.predict(test_set)
        logger_info.info('test data:')
        result_test = self.calculate_result(test_tag, test_pred)

        return dict({"train": result_train, "test": result_test})

    @fn_timer
    def labelPredict(self, doc_list):
        # 用训练好的分类器预测，返回最可能的结果
        clf_res = load(self.classify_path)
        tag_pred = clf_res.predict(doc_list)

        label_list = list()
        for tag in tag_pred:
            label = self.labels[tag]
            label_list.append(label)
        return label_list


    def labelPredictProb(self, doc_list, topN=4):
        # 用训练好的分类器预测,返回对应的结果topN的可能性
        clf_res = load(self.classify_path)
        tag_pred_prob = clf_res.predict_proba(doc_list)

        labels_list = list()
        for prob_list in tag_pred_prob:
            dic = dict()
            for index in range(len(prob_list)):
                dic[self.labels[index]] = float('%0.2f' % prob_list[index])
            labels_list.append(dic)

        labels_topN = list()
        for dic in labels_list:
            label_dict_count = sorted(dic.items(), key=lambda d: d[1], reverse=True)
            if topN > 0:
                labels_topN.append(label_dict_count[:topN])
            else:
                labels_topN.append(label_dict_count)
        return labels_topN


class TextProcessing(BaseFuction):
    """对文本进行预处理"""

    def __init__(self):
        super(TextProcessing, self).__init__()

    def file_train_model(self, texts):
        # 对用户输入的文件进行处理，用来训练模型
        with codecs.open(texts, 'r', encoding='utf-8', errors='ignore') as input_file, \
                codecs.open(self.cut_path, 'w', encoding='utf-8') as output_file:
            for line in input_file:
                content_cut = ' '.join(self.cut_jieba_stop(line))
                output_file.write(content_cut + '\n')


    def list_train_classify(self, texts, label_list, labels):
        # 对用户输入的列表进行处理，去训练分类器
        doc_list = list()
        tag_list = list()
        for text in texts:
            doc_list.append(' '.join(self.cut_jieba_stop(text)))
        for label in label_list:
            tag_list.append(labels.index(label))

        labels_str = ' '.join(labels)
        self.cf.set('labels_parameter','labels',labels_str)
        self.write_conf()

        return (doc_list, tag_list)

    def file_train_classify(self, file_path,labels):
        # 对用户输入的文件进行处理，用来训练分类器
        doc_list = list()
        label_list = list()
        tag_list = list()
        for root, dirs, files in os.walk(file_path):
            for file in files:
                input_file = codecs.open(os.path.join(root, file), 'r', encoding='utf-8')
                for line in input_file:
                    line_cut = ' '.join(self.cut_jieba_stop(line))
                    doc_list.append(line_cut)
                    label_list.append(file.split('.')[0])
                input_file.close()
        for label in label_list:
            tag_list.append(labels.index(label))

        labels_str = ' '.join(labels)
        self.cf.set('labels_parameter','labels',labels_str)
        self.write_conf()

        return doc_list, tag_list

    def list_label_predict(self, texts):
        # 对用户输入的列表进行处理，用来预测分类
        doc_list = list()
        for text in texts:
            doc_list.append(' '.join(self.cut_jieba_stop(text)))

        return doc_list

    def file_label_predict(self, file_path):
        # 对用户输入的文件进行处理，用来预测分类
        doc_list = list()
        for root, dirs, files in os.walk(file_path):
            for file in files:
                input_file = codecs.open(os.path.join(root, file), 'r', encoding='utf-8')
                for line in input_file:
                    line_cut = ' '.join(self.cut_jieba_stop(line))
                    doc_list.append(line_cut)
                input_file.close()

        return doc_list


