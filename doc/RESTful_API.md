# RESTful API

## 数据存储规范

训练时，数据存储在本地，通过POST方法传递数据的地址。客户需要提供如下的数据，并按照以下的格式存储数据。

- **root** 数据存储的根路径该路径下应用五个个文件夹corpus、label和unlabel。
- **root/corpus** 存放训练模型的大规模数据语料，系统默认提供，用户也可以自行上传训练语料，语料存放在一个文档中，文档格式为TXT，编码方式为UTF-8，数据分割符为换行符'\n'，文档中每一行是一条数据，数据条数不限。
- **root/label** 已标注数据存储的路径，该路径下有N个文档，N为需要分类的类别的个数，文档文件的名称为类别的名称，比如‘商业’， ‘娱乐’等，文档格式为TXT，编码方式为UTF-8，数据分割符为换行符'\n'，每篇文档中每一行是一条数据，数据条数不限。
- **root/unlabel** 未标注、待分类的数据存放的路径，该路径下存放为TXT文档，编码方式为UTF-8，数据分割符为换行符'\n'，文档中一行表示一篇未分类的数据。

**数据的路径存储在配置文件下`datasets_path`下**

## 顶层接口

- **描述**：将需要分类的文档已json的形式传递，返回分类结果。
- **URL** ：/labelPredictList/
- **示例**：

```json
// 输入json格式的分类信息
{
 "load_mode":0,//当flag为0时，load_mode为0表示直接用list的形式传递需要测试的文档数据，按照如下的格式传入texts。
 "texts":[  
	{"id":0,"content":"this is document one"},
	{"id":1,"content":"this is document two"},
	{"id":2,"content":"this is document ..."}					
],
  "labels":["商业","娱乐",...],
  "topN":1
}
```

```json
// 返回json信息(分类)
{
  "state":0,// state为0 表示分类预测完成,state为1 表示分类预测出错
  "task":"classification",
  "results":[
    {"id":1,"label":["商业"]},
    {"id":2,"label":["娱乐"]},
    {"id":3,"label":["..."]}
  ]
}
```

- **描述**：将需要分类的文档通过传递文件路径的形式传递，返回分类结果。
- **URL** ：/labelPredictFile/
- **示例**：

```json
// 输入json格式的分类信息
{
  "load_mode":1,//当flag为0时，load_mode为1表示给定需要测试的文档的路径，文件内一行表示一篇文档,文档之间用换行符'\n'分割，按照如下的格式。
  "load_addr":"./datasets/unlabel_texts/*.txt",
  "labels":["商业","娱乐",...]
}
```

```json
// 返回json信息(分类)
{
  "state":0, // state为0 表示分类预测完成
  "task":"classification",
  "results":[
    {"id":0,"label":["商业"]},
    {"id":1,"label":["娱乐"]},
    {"id":2,"label":["..."]}
  ]
 
}
```

- **描述**：用户以列表的形式直接给出标注数据，进行特征选择模型的训练，分类器模型的训练，以及模型保存。
- **URL** ：/classifyTrainList/
- **示例**：

```json
// 输入json格式的分类模型训练信息
{
  "load_mode":0,// 当flag为1时，load_mode为0表示直接用list的形式传递训练需要的已标注文档数据，按照如下的格式传入texts。
  "texts":[ 
	{"id":0,"label":"商业","content":"this is document one"},
	{"id":1,"label":"娱乐","content":"this is document two"},
	{"id":2,"label":"...","content":"this is document ..."}			
	],
  "labels":["商业","娱乐",...],
  "split_rate":0.8, // 分类train样本占总样本比率
  "n_feature_selection":2000, // tfidf向量特征选择的维度
  "classify_parameter":{
	"C":1, // 约束因子
	"kernel":"linear"// 核函数
  }
}
```

```json
// 返回json信息(训练分类器)
{
  "state":0, // state为0 表示分类器训练完成
  "task":"train classify",
  "train_accuracy":0.9,
  "test_accuracy":0.9,
}
```

- **描述**：用户以文件的形式给出标注数据，进行特征选择模型的训练，分类器模型的训练，以及模型保存。
- **URL** ：/classifyTrainFile/
- **示例**：


```json
// 输入json格式的分类模型训练信息
{
  "load_mode":1,// 当flag为1时，load_mode为1表示给定训练需要的已标注文档的路径，文档文件名表示类别名，文件内一行表示一篇文档,文档之间用换行符'\n'分割，按照如下的格式。
  "load_addr":"./datasets/label_texts/*.txt",
  "labels":["商业","娱乐",...]
}
```

```json
// 返回json信息(训练分类器)
{
  "state":0, // state为0 表示分类器训练完成
  "task":"train classify",
  "train_accuracy":0.9,
  "test_accuracy":0.9
}
```

- **描述**：用户自行导入训练语料，训练系统运行所需要的全部模型，包括词典模型的训练，tfidf模型的训练，lda模型的训练。（注：用户自行导入训练语料，训练模型，时间花费较长，不推荐。）
- **URL** ：/vectorizerTrain/
- **示例**：

```json
// 输入json格式的向量化模型训练信息
{
  "load_mode":1,// 当flag为2时，load_mode为1表示给定的大语料的路径，文件内一行表示一篇文档（无标注）,文档之间用换行符'\n'分割，按照如下的格式。
  "load_addr":"./datasets/corpus/*.txt"
}
```

```json
// 返回json信息(训练向量化模型)
{
  "state":0, // state为0 表示向量化模型训练完成
  "task":"train vectorization model"
  ]
}
```

## 设置配置文件

- **描述**：初始化接口，用来设置一些路径以及关键词等。配置文件的section包括：  
  - es:指定es的地址，端口等 
  - datasets_path:指定训练数据集的路径  
  - corpus_path:指定大规模语料的路径
  - vecter_path, classifier_path:指定向量化模型和分类器模型存储的地址  
- **URL** ：/setting/
- **示例**：

```json
// 输入json
{
  "Section":"vecter_path",
  "Option":"lda_model_path"
  "Value":"./user_models/lda/lda.model"
}
```

```json
// 返回json
{
    "state" : 1, //exit code 为 1 表示 配置失败
    "error info" : "section not exist"
}
```

```json
// 返回json
{
    "state" : 0 //exit code 为 0 表示 配置成功
}
```

## 使用说明

系统的所有功能都以RESTful API的形式发布，用户使用时，首先需要设定配置信息，包括：

- 修改ES配置文件，设置本机的IP和端口
- 设定训练数据的存放路径
- 设定模型的一些参数(可选，系统提供已经测试好的默认值)

其次，通过向`http://localhost:5000/`以`POST`的形式发送请求，进行**训练**。根据训练的返回信息可以确认是否完成全部的训练任务。如果需要查看具体的模块信息，可以到`info.log`文件查看。

最后，系统已经可以正式使用。用户只需要规定格式的json发送请求，即可得到每篇文档的分类结果。 
