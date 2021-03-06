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

from __future__ import print_function
import os
import warnings
import logging
import paddle
import paddle.fluid as fluid
import paddle.distributed.fleet.base.role_maker as role_maker
import paddle.distributed.fleet as fleet
import utils

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def get_reader(input_var, config):
    reader_type = config.get("static_benchmark.reader_type")
    train_data_path = config.get("static_benchmark.train_data_path")
    assert train_data_path != ""
    assert reader_type in ["QueueDataset", "DataLoader"]
    file_list = get_file_list(train_data_path, config)

    if reader_type == "QueueDataset":
        reader_instance = QueueDatset(input_var, file_list, config)
        return reader_instance.get_reader(), file_list
    elif reader_type == "DataLoader":
        reader_instance = DataLoader(input_var, file_list, config)
        return reader_instance.get_reader(), file_list


def get_infer_reader(input_var, config):
    test_data_path = config.get("static_benchmark.test_data_path")
    assert test_data_path != ""
    file_list = get_file_list(test_data_path, config)

    reader_instance = InferDataLoader(input_var, file_list, config)
    return reader_instance.get_reader(), file_list


def get_file_list(data_path, config):
    file_list = [
        data_path + "/%s" % x
        for x in os.listdir(data_path)
    ]
    if config.get("static_benchmark.split_file_list"):
        logger.info("Split file list for worker {}".format(fleet.worker_index(
        )))
        file_list = fleet.util.get_file_shard(file_list)
    logger.info("File list: {}".format(file_list))
    return file_list


def get_example_num(file_list):
    count = 0
    for f in file_list:
        last_count = count
        for _, _ in enumerate(open(f, 'r')):
            count += 1
        logger.info("File: %s has %s examples" % (f, count - last_count))
    logger.info("Total example: %s" % count)
    return count


def get_word_num(file_list):
    count = 0
    for f in file_list:
        last_count = count
        for index, line in enumerate(open(f, 'r')):
            line = line.rstrip().split()
            count += len(line)
        logger.info("file: %s has %s words" % (f, count-last_count))
    logger.info("Total words: %s" % count)
    return count


def get_reader_generator(path, reader_name="Reader"):
    reader_class = utils.lazy_instance_by_fliename(
        path, reader_name)()
    return reader_class


class DataLoader(object):
    def __init__(self, input_var, file_list, config):
        assert isinstance(input_var, list)
        assert len(file_list) > 0
        self.input_var = input_var
        self.file_list = file_list
        self.config = config

    def get_reader(self):
        logger.info("Get DataLoader")
        loader = fluid.io.DataLoader.from_generator(
            feed_list=self.input_var, capacity=64, iterable=False, use_double_buffer=False)
        path = self.config.get("static_benchmark.reader_path")
        generator = get_reader_generator(path)
        generator.init(self.config)
        batch_size = int(self.config.get("static_benchmark.batch_size"))
        loader.set_sample_generator(generator.dataloader(
            self.file_list), batch_size=batch_size, drop_last=True, places=paddle.static.cpu_places())
        return loader


class InferDataLoader(object):
    def __init__(self, input_var, file_list, config):
        assert isinstance(input_var, list)
        assert len(file_list) > 0
        self.input_var = input_var
        self.file_list = file_list
        self.config = config

    def get_reader(self):
        logger.info("Get DataLoader")
        loader = fluid.io.DataLoader.from_generator(
            feed_list=self.input_var, capacity=64, iterable=True)
        path = self.config.get("static_benchmark.reader_path")
        generator = get_reader_generator(path)
        generator.init(self.config)
        places = fluid.CPUPlace()
        loader.set_sample_generator(generator.dataloader(self.file_list),
                                    batch_size=int(self.config.get(
                                        "static_benchmark.batch_size")),
                                    drop_last=True,
                                    places=places)
        return loader


class QueueDatset(object):
    def __init__(self, input_var, file_list, config):
        assert isinstance(input_var, list)
        assert len(file_list) > 0
        self.config = config
        self.input_var = input_var
        self.file_list = file_list
        self.pipe_command = self.config.get("static_benchmark.pipe_command")
        assert self.pipe_command != None
        utils_path = utils.get_utils_file_path()
        self.pipe_command = "{} {} {}".format(
            self.pipe_command, config.get("yaml_path"), utils_path)
        self.batch_size = int(self.config.get("static_benchmark.batch_size"))
        assert self.batch_size >= 1
        self.thread_num = int(self.config.get("static_benchmark.thread_num"))
        assert self.thread_num >= 1

    def get_reader(self):
        logger.info("Get Dataset")
        dataset = fluid.DatasetFactory().create_dataset()
        dataset.set_use_var(self.input_var)
        dataset.set_pipe_command(self.pipe_command)
        dataset.set_batch_size(self.batch_size)
        dataset.set_thread(self.thread_num)
        dataset.set_filelist(self.file_list)
        return dataset
