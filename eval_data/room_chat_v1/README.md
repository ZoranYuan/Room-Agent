# Room Agent 单房间 RAG 评测数据 v1

## 内容

- `corpus.jsonl`：一个合成房间中的 4,000 条文本消息，`seq` 从 1 连续递增。
- `queries.jsonl`：100 个问题，其中 80 个可回答、20 个无答案；每题包含参考答案和 `gold_seqs`。
- `generate_dataset.py`：使用固定随机种子生成上述文件。
- `child_chunks.jsonl`：评测运行中由生产聚类和切分流程生成的子块快照，避免重复调用聚类模型。
- `retrieval_results.json`：逐题召回序号、分数、耗时及汇总指标。
- `run_retrieval_eval.py`：使用隔离的临时 Milvus 集合运行检索对比。

语料围绕 20 部电视剧或综艺展开，由 8 位虚构群成员讨论演员、作品、个人观感和观看安排；另穿插日常闲聊作为检索干扰。可回答问题各覆盖推荐者、推荐理由、具体看点和最后安排；无答案问题用于观察系统是否会在证据不足时拒答。

## 字段

消息记录兼容项目 `ChatMessage` 的消息字段，并额外带有 `room_id`、`type`。`send_time` 是为满足消息结构而生成的合成时间，不用于时间推理评测。

问题记录中的 `gold_seqs` 是支持答案的消息序号；无答案问题的 `answer` 为 `null`，`gold_seqs` 为空。

## 生成

```bash
python eval_data/room_chat_v1/generate_dataset.py
MILVUS_URI=http://milvus:19530 uv run --no-sync python eval_data/room_chat_v1/run_retrieval_eval.py
```

评测脚本创建隔离的临时 Milvus 集合，比较纯向量 Top-5、混合 RRF Top-5、混合 RRF 后重排 Top-3；结束后删除临时集合，结果写入 `retrieval_results.json`。需要可访问的 Milvus、MySQL 和项目配置的聚类模型服务。

## 评测边界

这是第一版合成 MVP，用于固定同一语料比较纯向量、混合检索和混合检索加重排。锚点事实由人工编写，闲聊由模板和词表组合生成；目前还不是人工撰写或真实用户聊天语料。正式报告应注明这一点，并在使用结果调参时另留未参与调参的问题集。
