"""统一定义异常消息、错误响应和失败日志文案。"""

DETAIL_MISSING_AUTH_TOKEN = "用户未认证"
DETAIL_IM_GRPC_AUTH_NOT_CONFIGURED = "IM 服务的 gRPC 身份校验尚未配置"
DETAIL_INVALID_AUTH_TOKEN = "认证令牌无效或已过期"
DETAIL_ROOM_ACCESS_DENIED = "无权访问该房间"

# 运行配置和 IM 响应。
DETAIL_MYSQL_NOT_CONFIGURED = "尚未配置 MYSQL_URL"
DETAIL_INVALID_MEMORY_BUDGET = "记忆窗口与摘要预算之和必须小于记忆压缩触发阈值"
DETAIL_IM_THREAD_ID_MISSING = "IM 鉴权响应缺少 Thread 标识"

# Agent 任务和流式事件。
DETAIL_CHAT_QUESTION_MISSING = "聊天任务缺少用户问题"
DETAIL_UNSUPPORTED_AGENT_TASK = "不支持的 Agent 任务类型: {task}"
DETAIL_CUSTOM_EVENT_NOT_DICT = "自定义事件数据必须是字典"
DETAIL_UNSUPPORTED_CUSTOM_EVENT = "不支持的自定义事件类型：{event_type!r}"
DETAIL_AGENT_RECURSION_LIMIT = "本次任务执行步骤过多，请缩小问题范围后重试。"
DETAIL_AGENT_TASK_FAILED = "Agent 任务执行失败"
DETAIL_AGENT_NO_MESSAGES = "Agent 未返回消息"
DETAIL_INVALID_TOOL_CALLS = "模型返回的工具调用参数无法解析"
DETAIL_AGENT_NO_FINAL_ANSWER = "Agent 未返回有效的最终回答"
DETAIL_MESSAGE_SEARCH_NOT_CONFIGURED = "消息检索服务尚未接入。"

# 历史记忆和消息聚类。
DETAIL_INVALID_MEMORY_SUMMARY = "会话记忆模型未返回有效的摘要消息"
DETAIL_MEMORY_SUMMARY_TRUNCATED = "会话摘要被输出长度限制截断，本次保留原始消息"
DETAIL_EMPTY_MEMORY_SUMMARY = "会话记忆模型返回空摘要"
DETAIL_INVALID_MESSAGE_GROUPS = "LLM 分组存在遗漏、重复或不存在的消息序列号"
DETAIL_INVALID_CLUSTER_RETRIES = "聚类重试次数配置无效"

# 父块存储。
DETAIL_CHUNK_ROOM_MISMATCH = "父块不属于当前房间"
DETAIL_INVALID_CHUNK_RANGE = "父块 seq 起始位置不能大于结束位置"

# 仅用于日志的失败说明。
LOG_AGENT_RECURSION_LIMIT = "Agent 达到最大执行步数"
LOG_AGENT_RUN_FAILED = "Agent 运行失败，请联系管理员"
LOG_MEMORY_COMPRESSION_FAILED = "会话记忆压缩失败，本轮临时采用最近完整对话"
LOG_MESSAGE_CLUSTER_RETRY = "消息聚类失败，准备第 %s 次重试"
