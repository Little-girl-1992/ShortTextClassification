# -*- coding: utf-8 -*-
"""
@author: lvrr
"""
import traceback
import logging
import codecs
import configparser
import json
import os

from flask import request, jsonify


from . import main
from .exe_code import EXE_Code


logger_info = logging.getLogger('info')
logger_error = logging.getLogger('error')
logger_time = logging.getLogger('time')


@main.route('/hello/')
def hello_world():
    print("Hello World!")
    return 'Hello World!'


@main.route('/labelPredictList/', methods=['POST'])
def labelPredictList():
    try:
        request_data = request.get_data().decode('utf-8')
    except Exception as e:
        logger_error.error(traceback.format_exc())
        return jsonify({"exit code": -1})

    try:
        json_request_data = json.loads(request_data, 'utf-8')
        load_mode = json_request_data['load_mode']
        texts_temp = json_request_data['texts']
        if type(texts_temp) == list:
            texts = [text for text in texts_temp]
        else:
            texts = [texts_temp]
        if 'topN' in json_request_data.keys():
            topN = json_request_data['topN']
        else:
            topN = -1
    except Exception as e:
        logger_error.error(traceback.format_exc())
        return jsonify({'exit code': -2})

    exe_code = EXE_Code()
    label_list = exe_code.online_classify(load_mode, texts, topN)

    return jsonify({"exit code":0,
                    "task":"classification",
                    "results":label_list
                    })


@main.route('/labelPredictFile/', methods=['POST'])
def labelPredictFile():
    try:
        request_data = request.get_data().decode('utf-8')
    except Exception as e:
        logger_error.error(traceback.format_exc())
        return jsonify({"exit code": -1})

    try:
        json_request_data = json.loads(request_data, 'utf-8')
        load_mode = json_request_data['load_mode']
        filename = json_request_data['load_addr']
        if 'topN' in json_request_data.keys():
            topN = json_request_data['topN']
        else:
            topN = -1
    except Exception as e:
        logger_error.error(traceback.format_exc())
        return jsonify({'exit code': -2})

    exe_code = EXE_Code()
    label_list = exe_code.online_classify(load_mode, filename, topN)

    return jsonify({"exit code":0,
                    "task":"classification",
                    "results":label_list
                    })


@main.route('/classifyTrainList/', methods=['POST'])
def classifyTrainList():
    try:
        request_data = request.get_data().decode('utf-8')
    except Exception as e:
        logger_error.error(traceback.format_exc())
        return jsonify({"exit code": -1})
    try:
        json_request_data = json.loads(request_data, 'utf-8')
        load_mode = json_request_data['load_mode']
        texts = [dic['content'] for dic in json_request_data['texts']]
        label_list = [dic['label'] for dic in json_request_data['texts']]
        labels = json_request_data['labels']
    except Exception as e:
        logger_error.error(traceback.format_exc())
        return jsonify({"exit code": -2})
    exe_code = EXE_Code()
    result_dict = exe_code.offline_train_classify(load_mode,labels, texts, label_list)
    if type(result_dict) == int and result_dict < 0:
        return jsonify({"exit code": result_dict})

    return jsonify({"exit code": 0,
                    "task": "train classify",
                    "train_accuracy": result_dict["train"],
                    "test_accuracy": result_dict["test"]
                    })


@main.route('/classifyTrainFile/', methods=['POST'])
def classifyTrainFile():
    try:
        request_data = request.get_data().decode('utf-8')
    except Exception as e:
        logger_error.error(traceback.format_exc())

        return jsonify({"exit code": -1})

    try:
        json_request_data = json.loads(request_data, 'utf-8')
        load_mode = json_request_data['load_mode']
        labels = json_request_data['labels']
        filename = json_request_data['load_addr']
        label_list = list()
    except Exception as e:
        return jsonify({"exit code": -2})

    exe_code = EXE_Code()
    result_dict = exe_code.offline_train_classify(load_mode, labels, filename, label_list)
    if type(result_dict) == int and result_dict < 0:
        return jsonify({"exit code": result_dict})

    return jsonify({"exit code":0,
                    "task":"train classify",
                    "train_accuracy":result_dict["train"],
                    "test_accuracy":result_dict["test"]
                    })


# 训练向量化模型
@main.route('/vectorizerTrain/', methods=['POST'])
def vectorizerTrain():
    try:
        request_data = request.get_data().decode('utf-8')
    except Exception as e:
        logger_error.error(traceback.format_exc())

        return jsonify({"exit code": -1})

    try:
        json_request_data = json.loads(request_data, 'utf-8')
        load_mode = json_request_data['load_mode']
        filename = json_request_data['load_addr']
    except Exception as e:
        logger_error.error(traceback.format_exc())
        return jsonify({'exit code': -2})

    exe_code = EXE_Code()
    state = exe_code.offline_train_model(load_mode, filename)
    if type(state)==int and state < 0:
        return jsonify({'exit code': state})

    return jsonify({"exit code": 0,
                    "task": "train classify"
                    })


@main.route('/setting/', methods=['POST'])
def setting():
    try:
        request_data = request.get_data().decode('utf-8')
    except Exception as e:
        logger_error.error(traceback.format_exc())
        return jsonify({"exit code": -1})

    try:
        json_request_data = json.loads(request_data, 'utf-8')
        section = json_request_data['section']
        option = json_request_data['option']
        value = json_request_data['value']
        cf = configparser.ConfigParser()
        cf.read(os.path.join(os.getcwd(),'app/main/initialization.conf'), encoding='utf-8')
        cf.set(section, option, value)
        with codecs.open(os.path.join(os.getcwd(),'app/main/initialization.conf'), 'w', encoding='utf-8') as configfile:
            cf.write(configfile)
    except Exception as e:
        logger_error.error(traceback.format_exc())
        jsonify({"exit code": -2})

    return jsonify({'exit code': 0})


