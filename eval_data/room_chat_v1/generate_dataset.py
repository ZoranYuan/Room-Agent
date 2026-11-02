"""Generate the deterministic Room Agent retrieval evaluation fixture."""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path


OUT_DIR = Path(__file__).parent
ROOM_ID = "eval-room-001"
CONVERSATION_ID = "eval-conversation-001"
SEED = 20261031
MESSAGE_COUNT = 4000

PEOPLE = [
    ("u01", "小周"),
    ("u02", "阿宁"),
    ("u03", "林可"),
    ("u04", "大伟"),
    ("u05", "小雨"),
    ("u06", "陈默"),
    ("u07", "阿哲"),
    ("u08", "佳佳"),
]

# 这些都是房间成员在语料里表达的观点或决定；评测依据是聊天记录本身，
# 不把这些主观内容当作现实世界的客观事实。
TOPICS = [
    {"title": "《开端》", "kind": "剧", "star": "白敬亭和赵今麦", "by": "小周", "reason": "公交车循环的设定很抓人，而且每次重来都会多一点新线索", "detail": "公交车上每个人的反应都不一样，配角也不像工具人", "decision": "周五晚上先看前三集，节奏合适再接着看"},
    {"title": "《狂飙》", "kind": "剧", "star": "张译和张颂文", "by": "阿宁", "reason": "人物关系不是简单的好人坏人，越往后越想知道他们怎么走到那一步", "detail": "她最记得菜市场那段对峙，几个人说话都留着劲", "decision": "这周先补前六集，周末再一起聊后面的反转"},
    {"title": "《繁花》", "kind": "剧", "star": "胡歌和马伊琍", "by": "林可", "reason": "画面和配乐很有年代感，台词也值得慢慢听", "detail": "她最喜欢黄河路夜景和饭店里那段谈生意的戏", "decision": "不赶进度，每晚看一集，先把人物关系理清楚"},
    {"title": "《庆余年》", "kind": "剧", "star": "张若昀和李沁", "by": "大伟", "reason": "权谋线里穿插很多轻松的梗，追起来不会太累", "detail": "他觉得范闲第一次进京都时和不同人的对话最好玩", "decision": "等第二季更新完再从第一季重看"},
    {"title": "《去有风的地方》", "kind": "剧", "star": "刘亦菲和李现", "by": "小雨", "reason": "节奏慢但看着舒服，云南的生活细节让她很想出去走走", "detail": "她最喜欢院子里大家一起吃饭、聊各自近况的片段", "decision": "留到假期看，顺便做一份大理旅行清单"},
    {"title": "《琅琊榜》", "kind": "剧", "star": "胡歌和王凯", "by": "陈默", "reason": "前后伏笔收得很整齐，重看时还能发现之前忽略的细节", "detail": "他觉得梅长苏和靖王在雪中谈话那场最有分量", "decision": "大家投票后决定从第一集开始重温"},
    {"title": "《漫长的季节》", "kind": "剧", "star": "范伟和秦昊", "by": "阿哲", "reason": "它把悬疑和普通人的生活放在一起，情绪留得很久", "detail": "他提到结尾那段向前走的画面，觉得克制但很有余味", "decision": "周末白天看完，晚上不接着刷短视频"},
    {"title": "《三体》", "kind": "剧", "star": "张鲁一和于和伟", "by": "佳佳", "reason": "她喜欢科学谜题一点点展开的过程，也想看群里怎么讨论设定", "detail": "她记得古筝行动那段大家看完都停了几秒才说话", "decision": "先看剧版，再找时间补原著前几章"},
    {"title": "《长相思》", "kind": "剧", "star": "杨紫和张晚意", "by": "小周", "reason": "人物之间的选择比较复杂，追剧时很适合边看边聊", "detail": "她觉得小夭和玱玹重逢时那段对话最让人纠结", "decision": "大家先看第一季，第二季等有空再补"},
    {"title": "《人世间》", "kind": "剧", "star": "雷佳音和殷桃", "by": "阿宁", "reason": "它从一家人的日常写时代变化，很多细节让人想到自己家里", "detail": "她印象最深的是周家人围桌吃饭、谁也没把话说满的那场戏", "decision": "挑几个周末慢慢看，不一次刷太多"},
    {"title": "《王牌对王牌》", "kind": "综艺", "star": "沈腾和贾玲", "by": "林可", "reason": "老搭档接梗很自然，游戏输了也能聊出笑点", "detail": "她提到猜歌环节里沈腾故意唱跑调，全场都接不上", "decision": "午饭时放一期，不专门熬夜追更新"},
    {"title": "《奔跑吧》", "kind": "综艺", "star": "李晨和郑恺", "by": "大伟", "reason": "户外游戏看着热闹，成员之间互相拆台挺有意思", "detail": "他最喜欢撕名牌前的推理环节，觉得比单纯跑更好看", "decision": "先挑有城市任务的那期看"},
    {"title": "《快乐再出发》", "kind": "综艺", "star": "陈楚生和苏醒", "by": "小雨", "reason": "大家熟了以后聊天很松弛，不用每分钟都安排一个游戏", "detail": "她最喜欢几个人在车里唱歌，唱完还认真讨论歌词", "decision": "做家务的时候从第一季第一期开始放"},
    {"title": "《你好，星期六》", "kind": "综艺", "star": "何炅和檀健次", "by": "陈默", "reason": "嘉宾多的时候互动变化快，何老师接话也很稳", "detail": "他提到猜身份游戏那期，檀健次几次差点说漏答案", "decision": "只看群里推荐的游戏环节，不追整季"},
    {"title": "《向往的生活》", "kind": "综艺", "star": "黄磊和何炅", "by": "阿哲", "reason": "做饭和聊天的节奏比较慢，适合下班后当背景音", "detail": "他记得大家在院子里边摘菜边聊小时候的事", "decision": "周日晚上看一集，配上自己煮的面"},
    {"title": "《声生不息》", "kind": "综艺", "star": "林子祥和叶蒨文", "by": "佳佳", "reason": "老歌新编能听出不同年代的味道，舞台也不只靠音量", "detail": "她最喜欢合唱《选择》那一段，觉得两个人的声音很搭", "decision": "把大家提到的歌曲先加进共享歌单"},
    {"title": "《脱口秀大会》", "kind": "综艺", "star": "徐志胜和呼兰", "by": "小周", "reason": "选手会把日常小事讲出新的角度，段子听完还能回味", "detail": "她觉得呼兰讲观察类段子时铺垫最完整", "decision": "从决赛那期开始看，再补喜欢的选手"},
    {"title": "《极限挑战》", "kind": "综艺", "star": "黄渤和黄磊", "by": "阿宁", "reason": "任务里有临场变化，嘉宾之间的反应比结果更好玩", "detail": "她最记得几个人为了线索在街区里问路，最后绕回原点", "decision": "找一集户外寻宝主题的重看"},
    {"title": "《圆桌派》", "kind": "综艺", "star": "窦文涛和马未都", "by": "林可", "reason": "聊天不急着下结论，能听到不同人的生活经验", "detail": "她喜欢聊旅行那期，嘉宾从行李聊到怎么面对陌生地方", "decision": "通勤时听音频版，回家再看完整视频"},
    {"title": "《乘风破浪》", "kind": "综艺", "star": "那英和周笔畅", "by": "大伟", "reason": "舞台编排能看出团队磨合，排练片段也挺有信息量", "detail": "他最喜欢几个人临时改和声后顺利合上的那次公演", "decision": "先看公演舞台合集，花絮留到周末"},
]


FILLER_TEMPLATES = [
    "我刚到{place}，今天{weather}，路上比平时{traffic}。",
    "午饭我点了{food}，本来怕{concern}，结果还挺{mood}。",
    "谁有{item}的推荐？我手里的这个最近{problem}。",
    "今天开会开到一半，突然想起{daily}，差点没接上同事的话。",
    "下班路上看到{scene}，莫名觉得今天也没那么糟。",
    "我把{task}拖到现在才做，果然还是得先列个小清单。",
    "周末如果不下雨，我想去{place}转一圈，不安排太满。",
    "刚才有人问我{question_topic}，我脑子里第一反应居然是{answer_topic}。",
    "这两天{drink}喝多了，晚上要换成热水。",
    "我发现{habit}这件事特别看心情，有时候十分钟就能搞定，有时候拖一整天。",
    "刚刷到一个讲{topic}的视频，评论区比视频本身还热闹。",
    "今天的{work}终于收尾了，剩下的明早再看，脑子已经不转了。",
    "你们吃过{food}吗？我之前一直觉得{concern}，今天试了下还不错。",
    "我妈又给我发了{media}，她每次都说这个一定要看完。",
    "等下我先去{errand}，回来再接着聊。",
    "刚才电梯里碰到邻居，聊了半天{daily}，差点坐过楼层。",
    "我给{item}充上电了，昨天忘了一晚上，难怪早上打不开。",
    "看天气预报说明天{weather}，带不带伞还挺纠结。",
    "我本来只想买{item}，最后又拎回来一袋{food}。",
    "今天最开心的事是{small_win}，虽然听起来不是什么大事。",
    "你们先聊，我这边{task}还差最后一步。",
    "刚才那段{topic}我没听清，等我回家把链接找出来。",
    "晚上不想点外卖了，准备把冰箱里的{food}解决掉。",
    "我最近在试着{habit}，目前坚持了几天，还不敢立 flag。",
    "谁记得上次我们说的{daily}？我只记得好像还没定时间。",
    "今天地铁上特别挤，我站了几站才找到能靠一下的位置。",
    "我已经把{task}写进待办了，不然转头肯定忘。",
    "这个周末想留半天什么都不干，剩下半天再看情况。",
    "刚才看到{scene}，突然想起以前也发生过差不多的事。",
    "我先不发表意见，等把{topic}看完再回来补课。",
]

FILLER_VALUES = {
    "place": ["公司楼下", "地铁口", "便利店", "小区公园", "咖啡店", "菜市场", "图书馆", "公交站"],
    "weather": ["风挺大", "有点闷", "突然降温", "太阳特别晒", "飘了会儿雨", "凉快不少"],
    "traffic": ["慢", "顺", "堵", "安静", "折腾"],
    "food": ["番茄鸡蛋面", "牛肉粉", "烤红薯", "凉面", "馄饨", "咖喱饭", "三明治", "豆花"],
    "concern": ["会太咸", "吃不饱", "踩雷", "有点腻", "不太新鲜"],
    "mood": ["舒服", "清爽", "满足", "惊喜", "合口味"],
    "item": ["耳机", "保温杯", "充电线", "雨伞", "键盘", "背包", "台灯", "水壶"],
    "problem": ["接触不太好", "续航变差", "有点松", "总是找不到", "开始掉漆"],
    "daily": ["买菜", "搬家", "订餐", "通勤", "健身", "整理房间", "约时间", "收快递"],
    "scene": ["小朋友追着鸽子跑", "路边摊排了长队", "一只猫趴在快递箱上", "有人在广场练舞", "晚霞刚好落在楼顶"],
    "task": ["改表格", "回邮件", "整理照片", "收拾桌面", "做预算", "写周报", "预约体检"],
    "question_topic": ["附近哪家店好吃", "周末去哪走走", "怎么整理照片", "最近有没有好歌"],
    "answer_topic": ["先把冰箱清空", "去河边走一走", "按月份建文件夹", "把老歌重新听一遍"],
    "drink": ["咖啡", "奶茶", "气泡水", "柠檬茶", "浓茶"],
    "habit": ["睡前少看手机", "每天走八千步", "周日提前备菜", "把零钱记账", "早上先晒太阳"],
    "topic": ["收纳", "城市散步", "耳机", "做饭", "通勤", "摄影", "露营", "老歌"],
    "work": ["报表", "需求文档", "排期", "会议纪要", "代码评审", "活动方案"],
    "media": ["短视频", "老照片", "菜谱", "新闻链接", "歌单"],
    "errand": ["拿快递", "买水果", "取干洗", "给猫买粮", "还书"],
    "small_win": ["把拖了很久的文件归档了", "终于找到那根线", "午休睡了二十分钟", "没错过公交"],
}


def _question_set(topic: dict[str, str], topic_id: int) -> list[dict[str, object]]:
    title = topic["title"]
    return [
        {"question": f"群里是谁明确推荐大家看{title}的？", "answer": topic["by"], "fact": "recommend"},
        {"question": f"推荐{title}时，对方主要看中了什么？", "answer": topic["reason"], "fact": "reason"},
        {"question": f"关于{title}，群里有人特别记得哪个片段或看点？", "answer": topic["detail"], "fact": "detail"},
        {"question": f"大家最后打算怎么安排{title}？", "answer": topic["decision"], "fact": "decision"},
    ]


def generate() -> None:
    rng = random.Random(SEED)
    rows: list[dict[str, object]] = []
    qa_rows: list[dict[str, object]] = []

    def add_message(sender: tuple[str, str], content: str) -> None:
        seq = len(rows) + 1
        rows.append(
            {
                "message_id": f"mock-msg-{seq:06d}",
                "room_id": ROOM_ID,
                "conversation_id": CONVERSATION_ID,
                "sender_id": sender[0],
                "sender_name": sender[1],
                "seq": seq,
                "type": 0,
                "content": content,
                "status": 0,
                "send_time": int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp()) + seq * 45,
            }
        )
    filler_index = 0

    def add_filler() -> None:
        nonlocal filler_index
        template = FILLER_TEMPLATES[filler_index % len(FILLER_TEMPLATES)]
        values = {key: rng.choice(value) for key, value in FILLER_VALUES.items()}
        content = template.format(**values)
        if filler_index % 17 == 3:
            content += rng.choice([" 哈哈", "😂", "我真的服了", "（捂脸）", "先记下来"])
        elif filler_index % 29 == 7:
            content = rng.choice(["嗯嗯，", "对，我也觉得。", "等下我看一下。", "收到。", "笑死。", "确实。"])
        sender = PEOPLE[(filler_index * 7 + rng.randrange(len(PEOPLE))) % len(PEOPLE)]
        add_message(sender, content)
        filler_index += 1

    # 均匀插入 20 段有明确证据的娱乐话题，避免 gold 证据都挤在语料开头。
    event_positions = {round(i * (MESSAGE_COUNT - 1) / len(TOPICS)) for i in range(len(TOPICS))}
    topic_cursor = 0
    while len(rows) < MESSAGE_COUNT:
        if topic_cursor < len(TOPICS) and len(rows) in event_positions:
            topic_id = topic_cursor
            topic = TOPICS[topic_id]
            recommender_person = next(person for person in PEOPLE if person[1] == topic["by"])
            others = [person for person in PEOPLE if person[1] != topic["by"]]
            opener, recommender, asker, detail_speaker, planner = [
                recommender_person,
                recommender_person,
                rng.choice(others),
                rng.choice(others),
                rng.choice(others),
            ]
            topic_evidence: dict[str, int] = {}
            add_message(rng.choice(others), f"最近有人补{topic['title']}吗？我看到好几次有人提。")
            add_message(opener, f"我刚好想聊{topic['title']}，主演是{topic['star']}，你们觉得值得补吗？")
            add_message(recommender, f"我推荐{topic['title']}，最近正好想找一部能和大家一起聊的。")
            topic_evidence["recommend"] = len(rows)
            add_message(asker, f"那你最记得{topic['title']}里的什么？")
            add_message(detail_speaker, f"我记得{topic['title']}里，{topic['detail']}。")
            topic_evidence["detail"] = len(rows)
            add_message(rng.choice(others), f"听着不像只靠热搜撑起来的，确实可以看看。")
            add_message(planner, f"那{topic['title']}我们怎么安排？最近大家时间不太一样。")
            add_message(planner, f"关于{topic['title']}，我建议{topic['decision']}。")
            topic_evidence["decision"] = len(rows)
            add_message(rng.choice(others), f"好，按这个来。我先把{topic['title']}加到待看清单。")

            # 分开给出推荐理由，问题与证据的表述不完全照抄。
            reason_sender = recommender
            add_message(rng.choice(others), f"你刚才说推荐{topic['title']}，吸引你的点具体是哪儿？")
            add_message(reason_sender, topic["reason"] + "，这点比单纯的明星阵容更打动我。")
            topic_evidence["reason"] = len(rows)
            add_message(rng.choice(others), "先记下来，等看过再回来对答案。")

            for qa in _question_set(topic, topic_id):
                evidence = topic_evidence[str(qa["fact"])]
                qa_rows.append(
                    {
                        "qid": f"q-{len(qa_rows) + 1:03d}",
                        "room_id": ROOM_ID,
                        "question": qa["question"],
                        "answerable": True,
                        "answer": qa["answer"],
                        "gold_seqs": [evidence],
                        "category": str(qa["fact"]),
                        "source": "synthetic_anchor",
                    }
                )
            topic_cursor += 1
        else:
            add_filler()

    # 20 个无法从聊天中回答的问题，用于测拒答；避免询问语料中已明确说出的信息。
    absent_questions = [
        ("《开端》在群里提到的导演生日是哪一天？", "unanswerable"),
        ("群里有人说《狂飙》一共拍了多少天吗？", "unanswerable"),
        ("《繁花》在群里提到的拍摄地具体是哪条街？", "unanswerable"),
        ("大家有说《庆余年》第二季的片尾曲叫什么吗？", "unanswerable"),
        ("群里提到《去有风的地方》是哪年开机的吗？", "unanswerable"),
        ("有人说《琅琊榜》原著一共多少章吗？", "unanswerable"),
        ("《漫长的季节》在群里的具体开播日期是哪天？", "unanswerable"),
        ("群里有人提到《三体》剧版每集的制作预算吗？", "unanswerable"),
        ("大家有说《长相思》第二季一共有多少集吗？", "unanswerable"),
        ("《人世间》里周家住址的门牌号，群里提过吗？", "unanswerable"),
        ("《王牌对王牌》那期猜歌环节的完整歌单是什么？", "unanswerable"),
        ("群里推荐的《奔跑吧》寻宝特辑具体是哪一季第几期？", "unanswerable"),
        ("《快乐再出发》车里唱歌那段是谁先提议的？", "unanswerable"),
        ("《你好，星期六》猜身份游戏的嘉宾名单完整是什么？", "unanswerable"),
        ("《向往的生活》院子里吃饭那期播出是哪一天？", "unanswerable"),
        ("《声生不息》群里说的《选择》是哪位编曲老师改编的？", "unanswerable"),
        ("《脱口秀大会》决赛的现场观众有多少人？群里提到了吗？", "unanswerable"),
        ("《极限挑战》寻宝那期最后的奖品是什么？", "unanswerable"),
        ("《圆桌派》旅行那期的录制地点具体在哪个城市？", "unanswerable"),
        ("《乘风破浪》那次公演的舞台灯光是谁设计的？", "unanswerable"),
    ]
    for question, category in absent_questions:
        qa_rows.append(
            {
                "qid": f"q-{len(qa_rows) + 1:03d}",
                "room_id": ROOM_ID,
                "question": question,
                "answerable": False,
                "answer": None,
                "gold_seqs": [],
                "category": category,
                "source": "synthetic_unanswerable",
            }
        )

    assert len(rows) == MESSAGE_COUNT
    assert len(qa_rows) == 100
    assert len({int(row["seq"]) for row in rows}) == MESSAGE_COUNT
    assert all(seq <= MESSAGE_COUNT for qa in qa_rows for seq in qa["gold_seqs"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "corpus.jsonl").open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (OUT_DIR / "queries.jsonl").open("w", encoding="utf-8") as file:
        for row in qa_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"messages={len(rows)} queries={len(qa_rows)} answerable=80 unanswerable=20")


if __name__ == "__main__":
    generate()
