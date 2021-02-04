#!/bin/bash
echo "WARNING: This script only for run Paddle Paddle CTR distribute training locally"

if [ ! -d "./log" ]; then
  mkdir ./log
  echo "Create log floder for store running log"
fi

# kill existing server process
ps -ef|grep python|awk '{print $2}'|xargs kill -9

# environment variables for fleet distribute training
export PADDLE_WITH_GLOO=0
export CPU_NUM=16

export PADDLE_PSERVER_NUMS=2
export PADDLE_TRAINERS_NUM=2

export PADDLE_PSERVERS_IP_PORT_LIST="127.0.0.1:6170,127.0.0.1:6171"
export PADDLE_PSERVER_PORT_ARRAY=( 6170 6171 )
export PADDLE_TRAINER_ENDPOINTS="127.0.0.1:6172,127.0.0.1:6173"


export TRAINING_ROLE=PSERVER
export GLOG_v=0
export GLOG_logtostderr=1
for((i=0;i<$PADDLE_PSERVER_NUMS;i++))
do
   cur_port=${PADDLE_PSERVER_PORT_ARRAY[$i]}
   echo "PADDLE WILL START PSERVER "$cur_port
   export PADDLE_PORT=${cur_port}
   export POD_IP="127.0.0.1"
   python -u ../train.py -c benchmark.yaml  &> ./log/pserver.$i.log &
done

export TRAINING_ROLE=TRAINER
export GLOG_v=0
export GLOG_logtostderr=1

for((i=0;i<$PADDLE_TRAINERS_NUM;i++))
do
   echo "PADDLE WILL START Trainer "$i
   export PADDLE_TRAINER_ID=$i
   python -u ../train.py -c benchmark.yaml &> ./log/trainer.$i.log &
done
