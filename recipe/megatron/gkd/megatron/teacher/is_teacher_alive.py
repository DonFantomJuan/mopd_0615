import zmq
from transformers import AutoTokenizer
from utils import serialize, deserialize

ckpt_path = "/home/ma-user/work/nlp/c00931509/code/verl/tmp/"
proxy_addr = "tcp://172.16.14.217:15555"

# 1) 加载 tokenizer
# 1) Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)

# 2) 你的问题
# 2) Your question
user_query = "你好，请简单介绍一下你自己。"

# 3) 用 chat template 构造标准对话输入
# 3) Build a standard chat prompt with the chat template
messages = [
    {"role": "user", "content": user_query}
]

try:
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
except Exception:
    # 如果 tokenizer 不支持 chat template，就手动退化成问答格式
    # If the tokenizer does not support chat templates, fall back to a manual Q&A format
    prompt_text = f"用户：{user_query}\n助手："

print("===== Prompt Text =====")
print(prompt_text)

# 4) 编码成 token ids
# 4) Encode into token ids
prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

# 注意这里要包成二维列表，外层是 batch
# Note that this must be wrapped as a 2D list, where the outer dimension is the batch
prompt_token_ids = [prompt_ids]

# 5) 组织请求
# 5) Build the request
req = {
    "prompt_token_ids": prompt_token_ids,
    "temperature": 0.8,
    "max_tokens": 4096,
    "only_response": False,
}

# 6) 连接 ZMQ teacher proxy
# 6) Connect to the ZMQ teacher proxy
ctx = zmq.Context()
sock = ctx.socket(zmq.REQ)
sock.connect(proxy_addr)

# 7) 发送请求并接收结果
# 7) Send the request and receive the result
sock.send(serialize(req))
resp = sock.recv()
data = deserialize(resp)

print("\n===== Raw Result =====")
print(data)

# 8) 解码模型返回的 token
# 8) Decode the returned tokens
if data.get("status") == "ok":
    response_ids = data["responses"][0].tolist()
    decoded_response = tokenizer.decode(response_ids, skip_special_tokens=True)

    print("\n===== Decoded Response =====")
    print(decoded_response)
else:
    print("\n===== Error =====")
    print(data.get("reason"))