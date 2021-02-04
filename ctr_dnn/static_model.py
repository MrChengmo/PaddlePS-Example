# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import paddle
import paddle.fluid as fluid
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
import math


class Model(object):
    """
    DNN for Click-Through Rate prediction
    """

    def __init__(self, config):
        self.cost = None
        self.metrics = {}
        self.config = config
        self.init_hyper_parameters()

    def init_hyper_parameters(self):
        self.dense_feature_dim = self.config.get(
            "hyper_parameters.dense_feature_dim")
        self.sparse_feature_dim = self.config.get(
            "hyper_parameters.sparse_feature_dim")
        self.embedding_size = self.config.get(
            "hyper_parameters.embedding_size")
        self.fc_sizes = self.config.get(
            "hyper_parameters.fc_sizes")

        self.learning_rate = self.config.get(
            "hyper_parameters.optimizer.learning_rate")
        self.adam_lazy_mode = self.config.get(
            "hyper_parameters.optimizer.adam_lazy_mode")

    def input_data(self):
        dense_input = fluid.layers.data(name="dense_input",
                                        shape=[self.dense_feature_dim],
                                        dtype="float32")

        sparse_input_ids = [
            fluid.layers.data(name="C" + str(i),
                              shape=[1],
                              lod_level=1,
                              dtype="int64") for i in range(1, 27)
        ]

        label = fluid.layers.data(name="label", shape=[1], dtype="int64")

        inputs = [dense_input] + sparse_input_ids + [label]
        return inputs

    def net(self, input):
        def embedding_layer(input):
            return fluid.layers.embedding(
                input=input,
                is_sparse=True,
                size=[self.sparse_feature_dim,  self.embedding_size],
                param_attr=fluid.ParamAttr(
                    name="SparseFeatFactors",
                    initializer=fluid.initializer.Uniform()),
            )

        dense_inputs = input[0:1]
        sparse_inputs = input[1:-1]
        label = input[-1]
        sparse_embed_seq = list(map(embedding_layer, sparse_inputs))

        concated = fluid.layers.concat(
            sparse_embed_seq + dense_inputs, axis=1)

        
        fc_input = [concated]
        fc_output = []
        for index, layer_size in enumerate(self.fc_sizes):
            fc_out = fluid.layers.fc(
                input=fc_input[-1],
                size=layer_size,
                act="relu",
                param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                    scale=1 / math.sqrt(fc_input[-1].shape[1]))), 
                name = "fc_{}".format(index)
            )
            fc_output.append(fc_out)
            fc_input.append(fc_out)

        pred = fluid.layers.fc(
                input=fc_output[-1],
                size=2,
                act="softmax",
                param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                    scale=1 / math.sqrt(fc_output[-1].shape[1]))),
            )
        cost = fluid.layers.cross_entropy(input=pred, label=label)
        avg_cost = fluid.layers.reduce_mean(cost)
        auc_var, _, _ = fluid.layers.auc(input=pred,
                                            label=label,
                                            num_thresholds=2**12,
                                            slide_steps=20)
        self.infer_target_var = auc_var
        self.cost = avg_cost
        return {'cost': avg_cost, 'auc': auc_var}

    def minimize(self, strategy=None):
        optimizer = fluid.optimizer.Adam(
            self.learning_rate, lazy_mode=self.adam_lazy_mode)
        if strategy != None:
            optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(self.cost)