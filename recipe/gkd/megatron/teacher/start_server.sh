export PROXY_FRONTEND_PORT=15555
export PROXY_BACKEND_PORT=15556

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export EXP_NAME=${EXP_NAME:-gkd-qwen3.6a3b}
export TP_SIZE=8

BACKEND=vllm
CKPT_PATH="/home/ma-user/work/nlp/c00931509/code/verl/qwen_30b_a3b_0327/"

wait_server_ready() {
    server=$1
    ip=$2
    port=$3
    while true; do
        echo "wait $server server ready at $ip:$port..."
        result=`echo -e "\n" | telnet $ip $port 2> /dev/null | grep Connected | wc -l`
        if [ $result -eq 1 ]; then
            break
        else
            sleep 1
        fi
    done
}

ps -ef | grep "python proxy.py" | grep -v grep | awk -F ' ' '{print $2}' | xargs -r kill -9
ps -ef | grep "python worker.py" | grep -v grep | awk -F ' ' '{print $2}' | xargs -r kill -9

nohup python proxy.py &> ./proxy.log 2>&1 &

wait_server_ready proxy localhost $PROXY_BACKEND_PORT

echo "teacher proxy is ready"

nohup python3 worker.py --backend $BACKEND --tp-size $TP_SIZE --n-logprobs 32 --ckpt-path $CKPT_PATH  > ./worker.log 2>&1 &
# nohup python worker.py --backend $BACKEND --tp-size 1 --n-logprobs 32 --ckpt-path $CKPT_PATH &> ./worker.log &

echo "start teacher worker"

echo "teacher server is ready"