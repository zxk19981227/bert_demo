# bert预训练模型使用教程

## **任务描述**

本任务是经典的 Stanford Sentiment Treebank 数据集上的情感二分类任务。

数据集主要是一些英语语句以及其对应的情感标签。在本数据集中，标签0代表消极情感，标签1代表积极情感。该任务的主要目的是使用经Masked Language Model方式训练后的bert-mini模型，在该数据集上进行fine-tune,以较小训练代价获得相对较高的准确度。

为了说明预训练模型的优势所在，笔者将预训练模型与基于bilstm神经网络的直接训练方式进行对比，实验结果如下：

|模型名称|单个epoch训练时间|训练epoch|测试准确度|
| :---------: | :-------------------------: | :----------:| :------------:|
|bert-mini|49s|4|71.33%|
|bilstm|4min|7|73.81%|


**以下为实验教程**

## 1. 环境配置
+ 安装python，详细安装方式可参考 <https://blog.csdn.net/weixin_40844416/article/details/80889165>

   注意：切勿安装最新版

+ 安装完成后，在主目录下运行指令 `pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple `完成python相关包的安装

   * Installation

     `python==3.7.1`

      `transformers==4.10.0`

      `torch==1.7.1`

      `numpy==1.20.3`

      `package>=20.9`
   
+ 其他版本对应关系

   | pytorch版本   | python版本        |
   | ------------- | ----------------- |
   |  1.5及以上     | python>=3.5       |
   | 1.4.0         | 3.5<=python<=3.8  |
   | 1.3.1         | 3.5 <=python<=3.8 |
   | 1.1.0 - 1.3.0 | 3.5 <=python<=3.7 |

transformers要求torch>1.1.0,python >=3.6 或者tensorflow>=2.0，该版本可以通过在完成torch、python环境配置后直接使用`pip install transformers`进行自动配置。requirements.txt中配置为笔者实验成功版本。


## 2. 数据准备
+ 代码下载

   下载地址：<https://github.com/zxk19981227/bert_demo>

+ 数据下载

  * 百度云下载：

    链接：https://pan.baidu.com/s/1nZg9w-EmJj7b0q_ZuBmLlQ 
    提取码：abcd
  
  将`train.tsv`,`test.tsv`,`valid.tsv`三个文件放在主目录下
  
+ 模型下载地址:

    * 1.官方网址下载

        下载地址：<https://huggingface.co/google/bert_uncased_L-2_H-128_A-2/tree/main>
        
        官方网址下载后`pytorch_model.bin`与`flax_model.msgpack`名称可能会变为乱码，需要对其重新命名为其原本名称。

    * 2.百度云下载

        链接：<https://pan.baidu.com/s/1jOD42eU1d-96Eu5pM6rqRQ>，提取码：abcd

        获取模型后，需要将`pytorch_model.bin`,`flax_model.msgpack`,`vocab.txt`,`config.json` 都放在主目录下的pytorch_model目录下。

## 3. 运行方式

在主目录下执行命令：`python main.py`

程序将自动以默认参数配置执行当前程序。


## 4. 运行结果

笔者本地测试环境使用cpu为`intel i7 10750H 2.62GHz`配置环境下，完成一个epoch训练需要实践大约为49s。

经过4个epoch以后valid结果出现最佳结果73%。

---

运行样例如下：

`0%|          | 0/865 [00:00<?, ?it/s]`

`100%|██████████| 865/865 [00:49<00:00, 17.31it/s]`

`0%|          | 0/109 [00:00<?, ?it/s]`

`epoch 0 train loss f0.62 accuracy:66.71%`

`100%|██████████| 109/109 [00:00<00:00, 143.94it/s]`

`epoch 0 valid loss f0.60 accuracy:68.12%`
`100%|██████████| 865/865 [00:54<00:00, 15.96it/s]`
`0%|          | 0/109 [00:00<?, ?it/s]`
`epoch 1 train loss f0.57 accuracy:71.92%`
`100%|██████████| 109/109 [00:00<00:00, 136.24it/s]`

`epoch 1 valid loss f0.61 accuracy:66.86%`
`100%|██████████| 865/865 [00:52<00:00, 16.57it/s]`
`0%|          | 0/109 [00:00<?, ?it/s]`
`epoch 2 train loss f0.52 accuracy:76.14%`
`100%|██████████| 109/109 [00:00<00:00, 131.75it/s]`

`epoch 2 valid loss f0.57 accuracy:71.22%`

---

*（以上运行情况仅供参考）*

## 5. 参数更改

具体参数可以通过命令行传递，如：`python main.py --lr 1e-4`.

具体参数名称以及作用可以通过 `python main.py --help` 查看。
