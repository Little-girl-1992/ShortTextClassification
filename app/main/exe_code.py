import os
import codecs
import logging
import traceback
from .main_code import TextProcessing
from .main_code import TrainModel
from .main_code import Vectorizer
from .main_code import Classify

logger_info = logging.getLogger('info')
logger_error = logging.getLogger('error')
logger_time = logging.getLogger('time')


def filename_and_labels(file_path,labels):
    if len(labels)< 2:
        logger_error.error('labels len < 2')
        return -11 # 用户输入labels长度错误
    if len(set(labels))!= len(labels):
        logger_error.error('len(set(labels)) != len(labels)')
        return -12 # 用户输入labels中类别有重复
    if '' in labels:
        logger_error.error('labels have \'\' ')
        return -13 #用户输入labels中有空字符
    labels_temp = list()
    for root, dirs, files in os.walk(file_path):
        for file in files:
            labels_temp.append(file.split('.')[0])
    if len(labels_temp) != len(labels):
        logger_error.error('len(labels_temp) != len(labels)')
        return -14 # 用户传入文件与传入类别个数不符
    if set(labels_temp) != set(labels):
        logger_error.error('set(labels_temp) != set(labels)')
        return -15 # 用户传入的样本文件与给定的类别不符

    return 0

def texts_and_labels(texts, label_list, labels):
    if len(labels)< 2:
        logger_error.error('len(labels)< 2')
        return -11 # 用户输入labels长度错误
    if len(set(labels))!= len(labels):
        logger_error.error('len(set(labels))!= len(labels)')
        return -12 # 用户输入labels中类别有重复
    if '' in labels:
        logger_error.error('labels have \'\'')
        return -13 #用户输入labels中有空字符
    if len(texts) != len(label_list):
        logger_error.error('len(texts) != len(label_list)')
        return -16 # 用户输入label_list长度与texts长度错误
    if set(label_list)!=set(labels):
        logger_error.error('len(set(label_list))!=len(set(labels))')
        return -17 # 用户输入label_list与labels元素不匹配
    return 0


class EXE_Code(TextProcessing,TrainModel,Vectorizer,Classify):
    """代码 API，实现不同分类模块"""
    def __init__(self):
        super(EXE_Code, self).__init__()

    def offline_train_model(self, load_mode, texts):
        # 数据预处理
        if load_mode:
            if os.path.isfile(texts):
                try:
                    self.file_train_model(texts)
                except Exception as e:
                    logger_error.error(traceback.format_exc())
                    return -5 # 数据读取错误
            else:
                return -4 # 数据上传失败
        else:
            return -3 # 选择错误模式

        # 训练lda模型
        try:
            self.train_lda_model()
        except Exception as e:
            logger_error.error(traceback.format_exc())
            return -6  # 模型生成错误

        # 训练tfidf模型
        try:
            self.train_tfidf_model()
        except Exception as e:
            logger_error.error(traceback.format_exc())
            return -6  # 模型生成错误

        return 0


    def offline_train_classify(self, load_mode, labels, texts, label_list=[]):
        # 数据预处理
        if load_mode: # 文件形式
            if os.path.isdir(texts):
                result_v = filename_and_labels(texts, labels)
                if result_v < 0:
                    return result_v
                try:
                    doc_list, tag_list = self.file_train_classify(texts, labels)
                    if '' in doc_list:
                        return -7 # 数据中包含不能判别的句子
                except Exception as e:
                    logger_error.error(traceback.format_exc())
                    return -5 # 数据读取错误
            else:
                return -4 # 数据上传失败

        else: # 列表形式
            result_v = texts_and_labels(texts, label_list, labels)
            if result_v < 0:
                return result_v
            try:
                doc_list, tag_list = self.list_train_classify(texts, label_list, labels)
                if '' in doc_list:
                    return -7 # 数据中包含不能判别的句子
            except Exception as e:
                logger_error.error(traceback.format_exc())
                return -5 # 数据读取错误

        # 向量化
        try:
            train_set, test_set, train_tag, test_tag = self.split_data_random(doc_list, tag_list)
            if os.path.isfile(self.chi2_path):
                os.remove(self.chi2_path)
            self.train_chi2(train_set, train_tag)

            train_set_v = self.output_merge_vertor(train_set)
            test_set_v = self.output_merge_vertor(test_set)
        except Exception as e:
            logger_error.error(traceback.format_exc())
            return -8

        # 训练分类器
        try:
            result_dict = self.train_svm_classify(train_set_v, test_set_v, train_tag, test_tag)
            return result_dict
        except Exception as e:
            logger_error.error(traceback.format_exc())
            return -9


    def online_classify(self, load_mode, texts, topN=-1):
        # 数据预处理
        if not os.path.isfile(self.classify_path):
            return -18
        if topN < -1 or type(topN)!= int:
            return -19 # 输入不符合要求
        try:
            if load_mode:
                if os.path.isdir(texts):
                    doc_list = self.file_label_predict(texts)
                    if '' in doc_list:
                        return -7 # 数据中包含不能判别的句子
                else:
                    return -4 # 数据上传错误
            else:
                doc_list = self.list_label_predict(texts)
                if '' in doc_list:
                    return -7 # 数据中包含不能判别的句子
        except Exception as e:
            logger_error.error(traceback.format_exc())
            return -5 # 数据读取错误

        #向量化
        try:
            test_set_v = self.output_merge_vertor(doc_list)
        except Exception as e:
            logger_error.error(traceback.format_exc())
            return -8

        #分类预测
        try:
            if topN == 0:
                labels_topN = self.labelPredict(test_set_v)
                return labels_topN
            else:
                labels_topN = self.labelPredictProb(test_set_v, topN)
                return labels_topN
        except Exception as e:
            logger_error.error(traceback.format_exc())
            return -9




