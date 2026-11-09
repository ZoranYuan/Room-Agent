"""Generate a deterministic 10k-message temporal retrieval benchmark."""

from __future__ import annotations

import json
import random
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

OUT_DIR = Path(__file__).parent
SEED = 20261108
ROOM_ID = "eval-room-v4"
CONVERSATION_ID = "eval-conversation-v4"
MESSAGES_PER_PARENT = 50
DATES = [date(2026, 7, 20) + timedelta(days=21 * index) for index in range(5)]
TIMEZONE = ZoneInfo("Asia/Shanghai")

PEOPLE = [
    (f"u{index:03d}", f"群友{index:03d}") for index in range(1, 101)
]

# 同一主题在不同日期反复讨论，答案有变化；时间排序应帮助定位对应的一次讨论。
TEMPORAL_TOPICS: list[tuple[str, str, list[str]]] = [
    ("2027届秋招投递节奏", "秋招投递", [
        "先投少量匹配岗位验证简历，再按反馈调整。", "先完成目标公司的提前批，再补常规批次。",
        "每周固定投递一批，避免集中在截止前提交。", "优先投递截止日期近的岗位，并记录进度。",
        "先投递最想去的岗位，收到反馈后再改简历。",
    ]),
    ("2027届秋招项目经历", "项目经历", [
        "用项目结果说明个人贡献，技术栈放到项目描述里。", "突出遇到的问题、采取的方案和可验证结果。",
        "把项目拆成个人负责的模块，避免只写团队成果。", "删掉背景铺垫，先写项目解决了什么问题。",
        "补充压测条件和指标口径，让结果可以复核。",
    ]),
    ("2027届秋招算法复习", "算法复习", [
        "每天练两道题，重点复盘思路而不是追求题量。", "按数组、树、图分专题复习，错题隔天重做。",
        "先补常见数据结构，再练中等难度题目。", "每周做一次限时模拟，记录卡住的环节。",
        "把高频题整理成模板，面试前优先复习错题。",
    ]),
    ("2027届秋招后端面试", "后端面试", [
        "先复习 MySQL 索引和事务，再准备项目追问。", "把并发、缓存和消息队列结合项目讲清楚。",
        "围绕一次线上故障准备排查过程和结果。", "先梳理服务链路，再解释性能瓶颈怎么定位。",
        "用白板讲清设计取舍，不要只背概念定义。",
    ]),
    ("2027届秋招目标城市", "目标城市", [
        "先按岗位数量筛城市，再比较生活成本。", "把上海和杭州的岗位机会、租房预算放在一起看。",
        "优先考虑有目标业务团队的城市，不只看城市名气。", "先投递几个城市的岗位，用面试反馈再做选择。",
        "给通勤和租房设预算上限，避免只比较税前薪资。",
    ]),
    ("2027届秋招面试复盘", "面试复盘", [
        "每场结束后记录没答好的问题，当天补上。", "按项目、基础和表达三个维度整理反馈。",
        "把追问点补进项目文档，下一场前重新演练。", "请同学模拟追问，练习先给结论再解释。",
        "区分知识不会和表达不清，分别安排复习。",
    ]),
    ("2027届秋招网申材料", "网申材料", [
        "提前准备成绩单、证件照和项目链接，统一放在一个目录。", "每提交一家公司就登记岗位、截止时间和状态。",
        "先核对联系方式和附件格式，再批量填表。", "把常见开放题准备成短版本，按岗位再微调。",
        "为不同岗位准备不同简历版本，提交前检查文件名。",
    ]),
    ("2027届秋招实习转正", "实习转正", [
        "先和导师确认转正标准，再把工作拆成可交付目标。", "整理实习期间负责的模块和上线结果。",
        "每周同步进度，及时暴露依赖和风险。", "准备一次项目复盘，重点讲个人决策和改进。",
        "同时了解外部校招机会，不把计划押在单一路径上。",
    ]),
    ("2027届秋招薪资沟通", "薪资沟通", [
        "先了解岗位区间和城市成本，再给出合理范围。", "把期望薪资和岗位级别、工作内容一起沟通。",
        "先确认总包构成，再比较月薪、奖金和补贴。", "保留协商空间，不在信息不足时过早报底价。",
        "拿到书面 offer 后再核对试用期和发薪规则。",
    ]),
    ("2027届秋招作品集", "作品集", [
        "先放最能体现岗位能力的两个项目，保证链接可访问。", "每个项目补上架构图、运行方式和关键结果。",
        "删掉没有说明价值的截图，重点呈现个人实现。", "把 README 写成背景、方案、效果和限制四部分。",
        "面试前检查仓库权限、依赖和演示数据。",
    ]),
    ("基金定投计划", "基金定投", [
        "先设每月固定额度，不因为短期涨跌临时加仓。", "把应急资金和投资资金分开管理。",
        "每季度检查一次配置，不按每天涨跌频繁调整。", "记录买入理由和风险承受范围。",
        "先看费率和跟踪指数，再决定是否纳入观察。",
    ]),
    ("黄金ETF讨论", "黄金ETF", [
        "关注利率和回撤，不把短期上涨当成确定趋势。", "先小额观察，等波动收敛再考虑调整。",
        "把黄金仓位控制在预设比例内。", "对照基金费率和跟踪误差再做选择。",
        "先记下买入条件，达到条件再行动。",
    ]),
    ("周末短途旅行", "短途旅行", [
        "选高铁两小时内的目的地，周日留出返程缓冲。", "先订可取消的住宿，再看天气决定路线。",
        "把集合时间和车次发群里，避免分头找人。", "一天安排两个主要地点，别把行程排太满。",
        "人均预算先定下来，再决定是否租车。",
    ]),
    ("动画补番安排", "动画补番", [
        "每周先看三集，留时间讨论剧情。", "选一部完结番，不同时开太多新坑。",
        "周末集中补番，工作日只看一集。", "先从节奏轻松的作品开始，避免片单太长。",
        "看完一季再投票决定下一部。",
    ]),
    ("电影夜安排", "电影夜", [
        "先投票选片，周五晚上八点开始。", "选时长两小时以内的电影，结束后留时间聊天。",
        "提前确认字幕版本和播放平台。", "准备两部候选片，到场后再做最后决定。",
        "把集合时间定在七点半，先确认投影设备。",
    ]),
    ("游戏联机计划", "游戏联机", [
        "先统一游戏版本，周六晚上进语音频道。", "按到场顺序轮换队长，避免等人太久。",
        "先打一局热身，再决定要不要排位。", "提前发房间号，迟到的人直接加入下一局。",
        "控制在两小时内，结束后再约下一次。",
    ]),
    ("读书会选书", "读书会", [
        "选章节较短的书，每周讨论一个主题。", "先从三本候选里投票，不选太厚的版本。",
        "每人准备一个问题，避免变成轮流复述。", "把阅读范围提前发出，缺席的人也能跟上。",
        "先试运行两周，再决定是否固定每月一次。",
    ]),
    ("音乐现场计划", "音乐现场", [
        "先确认开票时间，设置提醒后再决定是否购买。", "比较场馆交通和票价，不只看座位区域。",
        "提前约好散场集合点，避免人多走散。", "先听一遍歌单，现场更容易跟上。",
        "确认退改规则后再付款。",
    ]),
    ("合租看房安排", "合租看房", [
        "先核对通勤时间和水电费用，再约线下看房。", "把押金、租期和维修责任写进合同。",
        "白天和晚上各看一次周边环境。", "先确认室友作息，再决定是否合租。",
        "拍下房屋现状，交接时留存记录。",
    ]),
    ("聚餐地点选择", "聚餐地点", [
        "选地铁站附近、能预约且有多人桌的店。", "先统计忌口，再根据人数订位。",
        "人均预算控制在一百元左右，提前确认套餐。", "把地址和集合时间发群里，迟到的人自行到店。",
        "准备两个备选地点，满位时直接切换。",
    ]),
]

FILLER_TOPICS = [
    ("通勤", ["地铁换乘", "共享单车", "雨伞", "晚高峰"]),
    ("做饭", ["电饭煲", "剩菜", "调味", "洗碗"]),
    ("运动", ["散步", "拉伸", "跑步鞋", "步数"]),
    ("音乐", ["耳机", "歌单", "现场录音", "音量"]),
    ("阅读", ["书签", "读书进度", "电子书", "书架"]),
    ("日常", ["快递", "桌面", "充电器", "日历提醒"]),
    ("拍照", ["夜景", "相册备份", "镜头", "光线"]),
    ("天气", ["降温", "下雨", "防晒", "周末预报"]),
]
FILLER_REPLIES = [
    "我也遇到过，后来先把步骤记下来。",
    "这个办法挺实用，我晚点也试一下。",
    "收到，我先按这个准备，有变化再说。",
    "确实容易忘，设个提醒会省心些。",
    "我之前没注意这个点，感谢提醒。",
]


def epoch(day: date, minute_offset: int) -> int:
    local = datetime.combine(day, time(9, 0), tzinfo=TIMEZONE)
    return int((local + timedelta(minutes=minute_offset)).timestamp())


def build() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    rng = random.Random(SEED)
    events: list[dict[str, object]] = []
    queries: list[dict[str, object]] = []

    for topic_index, (topic, label, answers) in enumerate(TEMPORAL_TOPICS):
        for episode, day in enumerate(DATES):
            answer = answers[episode]
            latest = episode == len(DATES) - 1
            question = (
                f"最近一次群里讨论{topic}时，大家给出的建议是什么？"
                if latest
                else f"{day.year}年{day.month}月{day.day}日群里讨论{topic}时，大家建议怎么做？"
            )
            hint = (
                {"mode": "latest"}
                if latest
                else {
                    "mode": "range",
                    "start_at": datetime.combine(day, time.min, tzinfo=TIMEZONE).isoformat(),
                    "end_at": datetime.combine(day, time.max, tzinfo=TIMEZONE).isoformat(),
                }
            )
            query = {
                "qid": f"time_{topic_index + 1:02d}_{episode + 1}",
                "question": question,
                "answer": answer,
                "answerable": True,
                "kind": "temporal",
                "topic": topic,
                "time_hint": hint,
                "target_event_id": f"time-{topic_index + 1:02d}-{episode + 1}",
            }
            queries.append(query)
            events.append({
                "event_id": query["target_event_id"],
                "kind": "temporal",
                "topic": topic,
                "label": label,
                "answer": answer,
                "day": day,
                "query": query,
                "sort_key": (day, topic_index, episode),
            })

    non_temporal_domains = [
        "Java后端项目", "MySQL索引", "Redis缓存", "消息队列", "接口压测",
        "简历排版", "算法练习", "面试表达", "实习周报", "开源项目",
        "校园招聘", "offer比较", "股票观察", "预算记账", "周末徒步",
        "城市骑行", "动画配乐", "漫画交换", "桌游聚会", "宿舍收纳",
    ]
    aspects = ["准备清单", "常见问题", "时间安排", "选择标准", "复盘方法"]
    suggestions = [
        "先列出目标和截止时间，再按优先级逐项完成。",
        "准备一个小规模验证，拿到反馈后再决定是否扩大。",
        "把关键步骤写进清单，完成后留一份记录方便复盘。",
        "先比较成本和收益，保留一个备选方案。",
        "约好负责人和检查时间，避免只讨论不落地。",
    ]
    for index in range(100):
        domain = non_temporal_domains[index // 5]
        aspect = aspects[index % 5]
        topic = f"{domain}的{aspect}"
        answer = suggestions[(index * 3 + index // 5) % len(suggestions)]
        day = date(2026, 7, 20) + timedelta(days=(index * 7) % 105)
        query = {
            "qid": f"plain_{index + 1:03d}",
            "question": f"群里关于{topic}的具体建议是什么？",
            "answer": answer,
            "answerable": True,
            "kind": "plain",
            "topic": topic,
            "target_event_id": f"plain-{index + 1:03d}",
        }
        queries.append(query)
        events.append({
            "event_id": query["target_event_id"],
            "kind": "plain",
            "topic": topic,
            "label": aspect,
            "answer": answer,
            "day": day,
            "query": query,
            "sort_key": (day, 100 + index, 0),
        })

    events.sort(key=lambda event: event["sort_key"])
    corpus: list[dict[str, object]] = []
    children: list[dict[str, object]] = []
    for parent_index, event in enumerate(events, start=1):
        parent_id = f"v4-parent-{parent_index:04d}"
        day = event["day"]
        assert isinstance(day, date)
        topic = str(event["topic"])
        answer = str(event["answer"])
        members: list[dict[str, object]] = []
        for offset, content in enumerate([
            f"最近又聊到{topic}，大家有什么建议？",
            answer,
            f"这个建议我记下了，{event['label']}这块就按这个思路继续。",
            "我也同意，之后有进展再回群里同步。",
            f"先把{topic}的结论整理下来，免得和其他安排混在一起。",
        ]):
            seq = len(corpus) + 1
            sender_id, sender_name = PEOPLE[(parent_index * 7 + offset * 13) % len(PEOPLE)]
            row = {
                "message_id": f"v4-m{seq:06d}",
                "room_id": ROOM_ID,
                "conversation_id": CONVERSATION_ID,
                "sender_id": sender_id,
                "sender_name": sender_name,
                "seq": seq,
                "type": 0,
                "content": str(content),
                "status": 0,
                "send_time": epoch(day, 10 + offset * 2),
            }
            corpus.append(row)
            members.append(row)

        filler_groups: list[list[dict[str, object]]] = [[], [], []]
        for filler_index in range(45):
            group_index = filler_index // 15
            filler_topic, words = FILLER_TOPICS[
                (parent_index + group_index) % len(FILLER_TOPICS)
            ]
            word = words[(parent_index + filler_index) % len(words)]
            content = f"{filler_topic}的{word}我最近也在留意。 {rng.choice(FILLER_REPLIES)}"
            seq = len(corpus) + 1
            sender_id, sender_name = rng.choice(PEOPLE)
            row = {
                "message_id": f"v4-m{seq:06d}",
                "room_id": ROOM_ID,
                "conversation_id": CONVERSATION_ID,
                "sender_id": sender_id,
                "sender_name": sender_name,
                "seq": seq,
                "type": 0,
                "content": content,
                "status": 0,
                "send_time": epoch(day, 30 + filler_index),
            }
            corpus.append(row)
            filler_groups[group_index].append(row)

        for chunk_index, group in enumerate([members, *filler_groups], start=1):
            start_time = min(int(row["send_time"]) for row in group)
            end_time = max(int(row["send_time"]) for row in group)
            if chunk_index == 1:
                child_topic = topic
                summary = f"讨论结论：{answer}"
            else:
                filler_topic = FILLER_TOPICS[
                    (parent_index + chunk_index) % len(FILLER_TOPICS)
                ][0]
                child_topic = f"{filler_topic}日常闲聊"
                summary = "；".join(
                    str(row["content"]) for row in group[:5]
                )
            child = {
                "id": f"v4-child-{parent_index:04d}-{chunk_index}",
                "text": f"主题：{child_topic}\n摘要：{summary}",
                "metadata": {
                    "parent_id": parent_id,
                    "room_id": ROOM_ID,
                    "conversation_id": CONVERSATION_ID,
                    "topic": child_topic,
                    "seqs": [str(row["seq"]) for row in group],
                    "participant_ids": sorted({str(row["sender_id"]) for row in group}),
                    "start_time": start_time,
                    "end_time": end_time,
                },
            }
            children.append(child)
            if chunk_index == 1:
                event["child_id"] = child["id"]
                event["evidence_message_id"] = str(members[1]["message_id"])

    for query in queries:
        event = next(item for item in events if item["event_id"] == query["target_event_id"])
        query["gold_evidence_groups"] = [{
            "group_id": f"{query['qid']}_e1",
            "message_ids": [event["evidence_message_id"]],
            "child_id": event["child_id"],
        }]
        query["expected_evidence_blocks"] = 1
        query["room_id"] = ROOM_ID

    if len(corpus) != 10_000 or len(queries) != 200 or len(children) != 800:
        raise ValueError((len(corpus), len(queries), len(children)))
    return corpus, queries, children


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    corpus, queries, children = build()
    for filename, rows in (
        ("corpus.jsonl", corpus),
        ("queries.jsonl", queries),
        ("child_chunks.jsonl", children),
    ):
        with (OUT_DIR / filename).open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("Generated 10,000 messages, 800 child chunks, and 200 labeled questions.")
    print("Questions: 100 temporal (20 recurring topics), 100 without time hints.")


if __name__ == "__main__":
    main()
