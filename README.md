# PaddlePS-Example
基于PaddlePaddle、PaddleRec、PaddleFleet 提供模型的分布式示例


# 基本设计
```shell
# 以下文件会测试完毕后会放入 PaddleRec/tools/static文件夹
.
├── README.md
├── program.py     # 模型组网相关分支代码
├── reader.py      # dataset、dataloader的封装，及数据量统计工具
├── train.py       # 运行入口
├── utils.py       # yaml解析工具, 环境判断工具
```

```shell
# 以下文件夹及内容copy自paddlerec/models，相关修改后续合入PaddleRec  
├── ctr_dnn        
```


# 运行方法

- 安装 paddlepaddle 1.8.5版本

- 运行命令

    在ctr_dnn目录下

    模拟分布式运行ctr

    ```shell
    sh local_cluster.sh
    ```

    运行infer

    ```shell
    python -u ../infer.py -c benchmark.yaml
    ```

# 注意事项

## yaml中workspace的配置

需注意运行python时的目录，与模型组网、reade所在目录的相对关系，配置workspace

若在根目录下运行，应当将模型的`benchmark.yaml`里的`workspace`调整为模型目录，如ctr模型，调整为

```yaml
workspace: ctr_dnn
```

若在模型目录下运行，将`workspace`调整为`./`

## 执行分布式训练

在每台机器上设置环境变量后，执行 `python -u train.py`