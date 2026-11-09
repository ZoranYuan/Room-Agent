"""Generate a deterministic chat-history retrieval benchmark (v5)."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

OUT_DIR = Path(__file__).parent
SEED = 20261109
ROOM_ID = "eval-room-v5"
CONVERSATION_ID = "eval-conversation-v5"
TIMEZONE = ZoneInfo("Asia/Shanghai")
PARENT_MESSAGE_LIMIT = 50
PARENT_CHAR_LIMIT = 2_000
CHILD_CHAR_LIMIT = 400
MESSAGE_COUNT = 10_000

PEOPLE = [
    (f"u{i:02d}", name)
    for i, name in enumerate(
        [
            "阿岚", "小满", "北北", "柚子", "阿澈", "林间风", "橘猫", "晚星", "小禾", "青禾",
            "阿哲", "沐沐", "小岛", "南风", "栗子", "白露", "阿宁", "七喜", "小鱼", "山茶",
            "阿远", "可乐", "小鹿", "星野", "团子", "阿夏", "木木", "小周", "云朵", "阿森",
            "小雨", "初一", "长安", "阿梨", "小北", "麦冬", "一一", "小唐", "晴天", "阿简",
            "小叶", "知秋", "阿铭", "桃桃", "小顾", "海盐", "阿木", "小宋", "月白", "小乔",
        ],
        start=1,
    )
]

# 每个话题在不同日期重复讨论，建议内容会变化，供时间检索对照。
TIME_TOPICS: list[tuple[str, str, list[str]]] = [
    ("《原神》深境螺旋配队", "游戏", [
        "先用草行久打下半，把高爆发队留给上半。", "这期先带钟离，生存压力比竞速更大。",
        "上半用冻结队更稳，下半再处理多波小怪。", "练度不够就把输出位换成治疗，别硬凹满星。",
        "先看敌人抗性再换队，别照搬上期阵容。",
    ]),
    ("《崩坏：星穹铁道》遗器取舍", "游戏", [
        "主词条先对，再考虑副词条，不必追求每件都毕业。", "速度鞋优先留给需要抢回合的辅助。",
        "双暴差一点可以先用套装效果补，不急着重刷。", "先把主力角色的遗器强化到十二级再决定去留。",
        "刷到速度和有效副词条就先停，体力留给新角色。",
    ]),
    ("《王者荣耀》新赛季上分", "游戏", [
        "双排先选能补位的组合，别两个人都抢输出位。", "先练两三个版本强势英雄，排位里少临时换人。",
        "连输两把就休息，状态差时继续排只会更急。", "开局先报清楚分路和召唤师技能，减少误会。",
        "优先拿熟练度高的英雄，阵容缺控制时再补功能位。",
    ]),
    ("《黑神话：悟空》隐藏支线", "游戏", [
        "先回头检查土地庙附近的岔路，很多线索不在主路上。", "打完头目先别急着离开，场景里还有可互动的物件。",
        "把关键道具留到支线对话后再用，能多开一段剧情。", "地图边缘也要绕一圈，容易漏掉小怪和宝箱。",
        "先完成前置对话再去挑战，不然支线入口会错过。",
    ]),
    ("《塞尔达传说：王国之泪》探索路线", "游戏", [
        "先开塔确认地形，再沿河谷找神庙会省不少路。", "遇到高处就先找附近的弹射装置，不要一直爬。",
        "把耐寒料理带够，雪山区域可以一次清完。", "先做主线拿到能力，再回来解机关更顺手。",
        "营地附近做标记，支线回来时不容易迷路。",
    ]),
    ("《英雄联盟》世界赛观赛", "游戏", [
        "重点看前十五分钟的视野和先锋交换，比分落后不等于没机会。", "先关注打野路线，团战前的资源控制更能说明局势。",
        "看阵容后期曲线，别只凭一波团战判断谁更强。", "把BP和选手近期英雄池一起看，比单看版本榜更准。",
        "关键局先记下龙魂和大龙时间线，复盘会清晰很多。",
    ]),
    ("《动物森友会》岛屿改造", "游戏", [
        "先把主路和居民区定下来，装饰可以之后慢慢补。", "花田留一块统一区域，浇水和杂交都方便。",
        "博物馆周围做成广场，岛上动线会自然很多。", "先用临时道路试布局，确认满意再换正式地砖。",
        "商店和机场之间留宽一点，来访时不容易堵。",
    ]),
    ("《葬送的芙莉莲》旅途主题", "动漫", [
        "我们觉得重点是勇者离开后，芙莉莲才慢慢理解时间和陪伴。", "先看她和费伦的相处变化，情绪线比战斗线更关键。",
        "辛美尔留下的小事不断回响，串起了后面很多选择。", "这段更像迟来的告别，不是单纯的冒险回忆。",
        "可以从不同角色对寿命的感受来聊，主题会更完整。",
    ]),
    ("《咒术回战》角色动机", "动漫", [
        "先看角色做选择时在保护谁，动机比台词里的立场更可靠。", "我们打算重看涩谷篇，前后反差能看清人物变化。",
        "别只按战力排角色，承担的代价也是理解动机的一部分。", "这条线里同伴关系的变化比单场胜负更重要。",
        "把早期伏笔和后续行动对照起来，很多决定就不突兀了。",
    ]),
    ("《排球少年！！》比赛复盘", "动漫", [
        "先看发球轮次怎么打乱一传，再看二传如何调整进攻。", "那场比赛的转折是替补上场后接发球稳定下来。",
        "我们认为关键不是单人得分，而是拦防开始形成连续配合。", "先复盘暂停前后的站位变化，节奏转折很明显。",
        "把自由人和主攻的防守分工理清，比分变化就好理解了。",
    ]),
    ("《间谍过家家》家庭关系", "动漫", [
        "约尔和阿尼亚的日常互动让这个家庭慢慢有了真实感。", "先看彼此隐瞒身份时仍然做出的照顾，主题就在这里。",
        "我们觉得喜剧外壳下写的是互相接纳，不只是任务。", "阿尼亚的视角很重要，她经常把大人的谎言变成连接。",
        "后面讨论时可以对照最初的任务目标和现在的相处方式。",
    ]),
    ("《蓝色监狱》选手成长", "动漫", [
        "先观察每个人如何找到自己的武器，不必只比较进球数。", "这次讨论认为团队配合和个人欲望的冲突最有意思。",
        "把几次失败后的选择连起来看，成长轨迹更清楚。", "我们准备按比赛顺序复盘，不从角色人气开始聊。",
        "关键是他开始主动创造机会，而不是等队友喂球。",
    ]),
    ("《迷宫饭》料理与冒险", "动漫", [
        "先聊食材来源和队伍分工，料理设定让冒险更有生活感。", "我们记得那次讨论认为，吃饭也是队伍建立信任的过程。",
        "把怪物生态和料理方法放一起看，世界观细节很多。", "不需要每道菜都找现实原型，设定内部自洽就够了。",
        "这段的重点是大家从嫌弃到主动帮忙处理食材。",
    ]),
    ("《进击的巨人》结局讨论", "动漫", [
        "先把角色各自知道的信息列出来，再讨论选择是否合理。", "我们觉得结尾留出的分歧比给单一答案更有讨论空间。",
        "可以从自由的代价切入，前后几季的主题能连起来。", "先区分角色意图和行动结果，讨论不会混成一团。",
        "那次大家主要争论的是因果链，不是简单评价谁对谁错。",
    ]),
    ("南城地铁延时运营消息", "新闻", [
        "先确认试行线路和末班车时间，周末出行再看是否适用。", "群里提醒先查官方公告，社交平台转发的站点名单有误。",
        "这项安排只在节假日前后试行，平日时刻表暂时没变。", "我们建议把换乘末班时间也一起核对，别只看单条线路。",
        "先收藏运营方的通知页面，后续调整以那里为准。",
    ]),
    ("台风蓝色预警出行安排", "新闻", [
        "出门前看本地预警更新，沿江步道先暂停安排。", "群里决定把线下聚会改到地铁站附近的室内场地。",
        "关注学校和场馆的临时通知，不只看天气软件图标。", "雨具准备之外也要留意公交改线消息。",
        "等风力预警解除后再约户外活动，不提前冒险。",
    ]),
    ("本地博物馆夜场开放", "新闻", [
        "预约名额要提前在官方小程序抢，现场不一定能补票。", "夜场先看临展时间，常设展关闭会比白天早。",
        "我们打算周五去，进馆后先看二楼的城市史展。", "带学生证也要确认优惠票的入场时段。",
        "馆方公告说入口改到东门，导航不要直接搜老地址。",
    ]),
    ("社区旧书交换活动", "新闻", [
        "活动按类别收书，教材和杂志要分开放，方便现场整理。", "群里约好带三本以内，剩下的交给主办方统一流转。",
        "先登记书名和品相，避免到场后重复登记。", "现场有儿童读物专区，大家可以把绘本带过去。",
        "活动结束后未交换的书会捐给社区阅览室。",
    ]),
    ("城市公共充电站扩建", "新闻", [
        "先看新站是否支持夜间停车，不能只按充电枪数量判断。", "群里提到老小区附近会新增慢充位，快充站仍在审批。",
        "我们建议关注施工公告，开放日期可能会分批调整。", "地图上标注的点位还不是全部可用，先查运营状态。",
        "如果夜间收费不同，最好把服务费和停车费一起比较。",
    ]),
    ("高校秋季招聘会安排", "新闻", [
        "先看参会企业名单和专业要求，再决定带几份简历。", "线上预约入口开在学校就业平台，现场也要带学生证。",
        "群里建议按目标岗位规划路线，别从头到尾随便逛。", "宣讲会时间撞车时，先去目标岗位更多的那场。",
        "把招聘会后的投递截止日期记下来，现场聊完及时跟进。",
    ]),
]

DATES = [
    date(2025, 11, 15),
    date(2026, 2, 15),
    date(2026, 6, 15),
    date(2026, 8, 15),
    date(2026, 10, 15),
]
TIME_EXPRESSIONS = ["去年11月15日", "今年2月15日", "2026年6月15日", "2026年8月15日", "上个月"]
PLAIN_FACETS = ["入门准备", "路线安排", "资源取舍", "组队协作", "复盘方式"]
PLAIN_ANSWERS = {
    "游戏": [
        "先把常用配置存好，再去测试高难内容。",
        "先清主线标记，再沿地图边缘补收集物。",
        "稀缺材料优先留给常用角色，不要平均分配。",
        "联机前先说清角色职责，避免重复选位。",
        "打完记下卡点和掉落，下一轮直接调整。",
    ],
    "动漫": [
        "按正片播放顺序补，特别篇放到对应季度后。",
        "先统一动画集数，再回头讨论设定细节。",
        "把片尾曲和插曲按集数记下来，之后更容易找到。",
        "讨论前先说清动画还是漫画进度，避免误剧透。",
        "按剧情顺序整理伏笔，不要只按角色人气归类。",
    ],
    "新闻": [
        "先找发布单位的原文，核对日期和适用范围。",
        "出门前看官方实时公告，再确认交通应用的变更。",
        "把预约入口和截止时间一起记进日历。",
        "有争议时把来源链接发群里，不转无出处截图。",
        "等正式通告更新后再调整安排，不按传言行动。",
    ],
}

FILLER = {
    "游戏": [
        "昨晚那把最后差一点，今天上线再试试。", "我先把更新包下好，免得约好时间还在等。",
        "手柄电量记得看一下，上次打到一半没电了。", "新活动我只做了日常，周末再清剩下的。",
        "这个设置可以关掉，画面会清爽不少。", "我把截图发群文件了，方便大家之后找。",
    ],
    "动漫": [
        "片尾曲挺适合循环，我已经加进歌单了。", "这周先补完手头这部，再开新的。",
        "平台字幕版本不太一样，最好确认一下集数。", "我先不看剧透，周末补完再回来讨论。",
        "大家发的截图我存起来了，后面找作画细节。", "这集节奏慢一点，但氛围还不错。",
    ],
    "新闻": [
        "这条先等官方更新，转发的截图看不出发布时间。", "我把来源链接放群里了，大家可以核对原文。",
        "路上施工绕行了，出门前最好看下实时地图。", "今天气温变化挺大，晚点回去记得带外套。",
        "社区公告栏也贴了通知，线上线下说法一致。", "标题有点夸张，正文里的适用范围比较窄。",
    ],
}

PLAIN_TOPICS = [
    ("《原神》枫丹水下探索", "游戏", "水下区域先开哪些传送点？", "先沿主线开中央区域，再补两侧洞窟的传送点。"),
    ("《原神》尘歌壶访客", "游戏", "怎么让朋友更容易找到壶里的摊位？", "把摊位放在入口右侧，用石板路连到主路。"),
    ("《星穹铁道》模拟宇宙祝福", "游戏", "打高难模拟宇宙优先拿什么祝福？", "先凑出能稳定生存的祝福，再补主要输出路径。"),
    ("《星穹铁道》忘却之庭排轴", "游戏", "忘却之庭总差一回合怎么调整？", "先把速度慢的辅助往前调，输出技能留给弱点击破后。"),
    ("《王者荣耀》游走位开局", "游戏", "辅助开局先跟谁比较合适？", "先帮中路抢线，再看打野准备从哪边开。"),
    ("《王者荣耀》对抗路传送", "游戏", "边路传送技能什么时候交？", "等小龙刷新前留着，支援完还能回线吃兵。"),
    ("《黑神话：悟空》装备搭配", "游戏", "打高韧性头目用什么思路配装？", "先保证棍势积累，再带一个能稳定打断的法术。"),
    ("《塞尔达传说：王国之泪》电池升级", "游戏", "前期电池不够用去哪里换？", "先把左纳乌能源带到监视堡垒的管家魔像处兑换。"),
    ("《艾尔登法环》法师开荒", "游戏", "法师刚开荒先补哪项能力？", "先把生命点到能扛住一套，再逐步补智力。"),
    ("《动物森友会》花卉杂交", "游戏", "怎么提高稀有花色出现的机会？", "先按棋盘格隔开同色花，每天浇水并记录新花位置。"),
    ("《葬送的芙莉莲》补番顺序", "动漫", "补番前需要先看特别篇吗？", "按正片顺序看就能理解，特别篇放在第一季之后即可。"),
    ("《咒术回战》动画与漫画衔接", "动漫", "第二季看完后从漫画哪里接？", "从涩谷事变后续章节开始，先回看最后一集对应的章节。"),
    ("《排球少年！！》剧场版", "动漫", "看剧场版前要补到哪一季？", "先看完第四季，再看垃圾场决战会更有代入感。"),
    ("《间谍过家家》角色关系", "动漫", "约尔的工作身份是哪一集讲清楚的？", "第一季前段就交代了，回看她第一次和黄昏见面的那集。"),
    ("《迷宫饭》漫画进度", "动漫", "动画完结后漫画从哪一话接？", "从动画最后一集覆盖章节的下一话接着看最稳妥。"),
    ("《蓝色监狱》角色入门", "动漫", "新观众先记住哪几个位置？", "先分清前锋、边锋和中场的职责，比赛场面就容易跟上。"),
    ("《进击的巨人》重看伏笔", "动漫", "重看时哪类细节最值得记？", "记下墙内外信息差和角色称呼变化，后面能对上不少伏笔。"),
    ("《紫罗兰永恒花园》观看体验", "动漫", "想看单元故事从哪几集开始？", "从第一集按顺序看，人物关系和情绪递进是连续的。"),
    ("《孤独摇滚！》乐队讨论", "动漫", "想入坑先听哪首现场曲？", "先听团结Band的主打曲，再看文化祭那段现场演出。"),
    ("《夏目友人帐》系列顺序", "动漫", "系列比较长应该怎么安排？", "按季度顺序看，特别篇放在对应季度之后。"),
    ("南城地铁新站开放", "新闻", "新站开通首周怎么查接驳公交？", "看交通局公告里的线路表，地图应用通常会晚几天更新。"),
    ("滨河步道临时封闭", "新闻", "滨河步道封闭期间从哪边绕行？", "从图书馆侧门转到北岸绿道，施工围挡外有指示牌。"),
    ("本地暴雨黄色预警", "新闻", "收到预警后晚高峰出行先确认什么？", "先看地铁和公交运营公告，再决定是否提前出门。"),
    ("市图书馆延长开放", "新闻", "周末夜读需要提前预约吗？", "夜间自习区要在公众号预约，普通阅览区按现场名额进。"),
    ("社区旧物回收日", "新闻", "旧电池和小家电能一起交吗？", "分开装袋，电池交专用回收箱，小家电交服务台登记。"),
    ("校园动漫展交通", "新闻", "去会展中心哪条地铁线少走路？", "坐二号线到会展中心站，从D口出来最近。"),
    ("本地青年影展排片", "新闻", "影展的映后交流怎么报名？", "先在主办方小程序选场次，映后名额和电影票分开预约。"),
    ("城市共享单车新规", "新闻", "新停车规则上线后怎么避免被多扣费？", "在电子围栏内还车，锁车后确认应用显示订单已结束。"),
    ("高校科技节报名", "新闻", "科技节的工作坊需要准备电脑吗？", "编程工作坊要自带电脑，其他展区不需要额外设备。"),
    ("社区宠物义诊活动", "新闻", "带猫去义诊前要准备什么资料？", "带疫苗记录和近期用药情况，现场按预约时段排队。"),
]

ANSWER_MESSAGES = [
    "我记下来了，按这个做应该比较稳。", "这个细节之前真没注意，回头试一下。",
    "我支持这个方案，至少步骤比较清楚。", "先这样安排，有结果再回来补充。",
    "明白了，我把链接和说明一起存下来。",
]
FOLLOWUPS = [
    "要是有人试过不同做法，也可以在群里补充。", "我晚点把相关截图整理好发出来。",
    "先按这个思路走，碰到例外再讨论。", "这条信息挺实用，我转给没赶上讨论的人。",
    "之后有更新的话记得把日期也带上，免得和旧消息混了。",
]


def epoch(day: date, minute: int) -> int:
    local = datetime.combine(day, time(9, 0), tzinfo=TIMEZONE)
    return int((local + timedelta(minutes=minute)).timestamp())


def message_text(row: dict[str, object]) -> str:
    return (
        f"[seq={row['seq']}, send_time={row['send_time']}] "
        f"[{row['sender_name']} ({row['sender_id']})]: {row['content']}"
    )


def make_query_time(topic: str, expression: str, episode: int) -> str:
    templates = [
        f"{expression}群里聊到{topic}时，大家建议先做什么？",
        f"回想{expression}关于{topic}的讨论，最后给出的办法是什么？",
        f"{expression}大家提到{topic}时，最重要的注意点是什么？",
        f"群里在{expression}讨论{topic}后，大家决定怎么安排？",
        f"{expression}聊{topic}那次，最后推荐的做法是什么？",
    ]
    return templates[episode]


def build() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    rng = random.Random(SEED)
    events: list[dict[str, object]] = []

    for topic_index, (topic, domain, answers) in enumerate(TIME_TOPICS):
        for episode, day in enumerate(DATES):
            event_id = f"time-{topic_index + 1:02d}-{episode + 1}"
            expression = TIME_EXPRESSIONS[episode]
            is_latest = episode == 4 and topic_index % 2 == 0
            question = (
                f"最近一次群里聊到{topic}时，大家建议先做什么？"
                if is_latest
                else make_query_time(topic, expression, episode)
            )
            hint_start = datetime.combine(day, time.min, tzinfo=TIMEZONE)
            hint_end = datetime.combine(day, time.max, tzinfo=TIMEZONE)
            if expression == "上个月":
                hint_start = datetime(2026, 10, 1, tzinfo=TIMEZONE)
                hint_end = datetime(2026, 10, 31, 23, 59, 59, 999999, tzinfo=TIMEZONE)
            query = {
                "qid": event_id,
                "question": question,
                "answerable": True,
                "has_time_intent": True,
                "kind": "temporal",
                "topic": topic,
                "time_hint": (
                    {"mode": "latest"}
                    if is_latest
                    else {
                        "mode": "range",
                        "start_at": hint_start.isoformat(),
                        "end_at": hint_end.isoformat(),
                    }
                ),
            }
            events.append({
                "event_id": event_id,
                "topic": topic,
                "domain": domain,
                "day": day,
                "answer": answers[episode],
                "query": query,
                "sort_key": (day, topic_index, episode),
            })

    plain_events: list[dict[str, object]] = []
    for index in range(150):
        base_topic, domain, _, _ = PLAIN_TOPICS[index % len(PLAIN_TOPICS)]
        cycle = index // len(PLAIN_TOPICS)
        facet = PLAIN_FACETS[cycle]
        topic = f"{base_topic}：{facet}"
        day = date(2026, 1, 5) + timedelta(days=(index * 3 + cycle * 11) % 290)
        question = f"关于{base_topic}，群里对{facet}的建议是什么？"
        answer = PLAIN_ANSWERS[domain][cycle]
        event_id = f"plain-fact-{index + 1:03d}"
        plain_events.append({
            "event_id": event_id,
            "topic": topic,
            "domain": domain,
            "day": day,
            "answer": answer,
            "query_text": question,
            "sort_key": (day, 100 + index, 0),
        })
        events.append(plain_events[-1])

    events.sort(key=lambda event: event["sort_key"])
    corpus: list[dict[str, object]] = []
    event_message_ids: dict[str, list[str]] = {}
    event_answer_message_id: dict[str, str] = {}
    filler_topic_by_id: dict[str, str] = {}

    def add_message(day: date, minute: int, content: str, *, event_id: str | None = None, filler_topic: str | None = None) -> dict[str, object]:
        seq = len(corpus) + 1
        sender_id, sender_name = PEOPLE[(seq - 1) % len(PEOPLE)]
        row: dict[str, object] = {
            "message_id": f"v5-m{seq:06d}",
            "room_id": ROOM_ID,
            "conversation_id": CONVERSATION_ID,
            "sender_id": sender_id,
            "sender_name": sender_name,
            "seq": seq,
            "type": 0,
            "content": content,
            "status": 0,
            "send_time": epoch(day, minute),
        }
        corpus.append(row)
        if event_id is not None:
            event_message_ids.setdefault(event_id, []).append(str(row["message_id"]))
        if filler_topic is not None:
            filler_topic_by_id[str(row["message_id"])] = filler_topic
        return row

    for event_index, event in enumerate(events):
        topic = str(event["topic"])
        domain = str(event["domain"])
        day = event["day"]
        assert isinstance(day, date)
        event_id = str(event["event_id"])
        answer = str(event["answer"])
        if event.get("query"):
            opener = f"刚才又有人提到{topic}，我想听听大家的看法。"
        else:
            opener = f"问一下，关于{topic}，之前有人试过吗？"
        contents = [
            opener,
            answer,
            rng.choice(ANSWER_MESSAGES),
            rng.choice(FOLLOWUPS),
            f"把这次关于{topic}的结论记一下，后面别和别的话题弄混。",
        ]
        rows = [
            add_message(day, 10 + offset * 3, content, event_id=event_id)
            for offset, content in enumerate(contents)
        ]
        event_answer_message_id[event_id] = str(rows[1]["message_id"])
        event["start_time"] = rows[0]["send_time"]
        event["end_time"] = rows[-1]["send_time"]

        # 每个重点讨论之间插入同领域闲聊，避免评测语料只由答案片段组成。
        filler_day = day
        for filler_index in range(35):
            content = rng.choice(FILLER[domain])
            add_message(
                filler_day,
                35 + filler_index * 2,
                content,
                filler_topic=f"{domain}闲聊",
            )

    # Parent blocks use the production batching rule: stop at either limit.
    parents: list[dict[str, object]] = []
    parent_for_message: dict[str, str] = {}
    current: list[dict[str, object]] = []
    current_chars = 0
    for row in corpus:
        row_chars = len(message_text(row))
        if current and (
            len(current) >= PARENT_MESSAGE_LIMIT
            or current_chars + row_chars > PARENT_CHAR_LIMIT
        ):
            parent_id = f"v5-parent-{len(parents) + 1:04d}"
            parents.append({
                "id": parent_id,
                "room_id": ROOM_ID,
                "conversation_id": CONVERSATION_ID,
                "seq_start": current[0]["seq"],
                "seq_end": current[-1]["seq"],
                "message_ids": [item["message_id"] for item in current],
                "message_count": len(current),
                "char_count": current_chars,
            })
            for item in current:
                parent_for_message[str(item["message_id"])] = parent_id
            current = []
            current_chars = 0
        current.append(row)
        current_chars += row_chars
    if current:
        parent_id = f"v5-parent-{len(parents) + 1:04d}"
        parents.append({
            "id": parent_id,
            "room_id": ROOM_ID,
            "conversation_id": CONVERSATION_ID,
            "seq_start": current[0]["seq"],
            "seq_end": current[-1]["seq"],
            "message_ids": [item["message_id"] for item in current],
            "message_count": len(current),
            "char_count": current_chars,
        })
        for item in current:
            parent_for_message[str(item["message_id"])] = parent_id

    message_by_id = {str(row["message_id"]): row for row in corpus}
    child_chunks: list[dict[str, object]] = []
    child_for_message: dict[str, str] = {}

    def add_child(parent_id: str, topic: str, summary: str, member_ids: list[str], child_id: str) -> None:
        members = [message_by_id[message_id] for message_id in member_ids]
        text = f"主题：{topic}\n摘要：{summary}\n" + "\n".join(
            f"[{item['sender_name']}，{item['send_time']}] {item['content']}"
            for item in members
        )
        child_chunks.append({
            "id": child_id,
            "text": text[:CHILD_CHAR_LIMIT],
            "metadata": {
                "parent_id": parent_id,
                "room_id": ROOM_ID,
                "conversation_id": CONVERSATION_ID,
                "topic": topic,
                "seqs": [str(item["seq"]) for item in members],
                "participant_ids": sorted({str(item["sender_id"]) for item in members}),
                "start_time": min(int(item["send_time"]) for item in members),
                "end_time": max(int(item["send_time"]) for item in members),
            },
        })
        for message_id in member_ids:
            child_for_message[message_id] = child_id

    # Gold semantic chunks are event-based and are created independently from queries.
    for event in events:
        event_id = str(event["event_id"])
        topic = str(event["topic"])
        member_ids = event_message_ids[event_id]
        by_parent: dict[str, list[str]] = defaultdict(list)
        for message_id in member_ids:
            by_parent[parent_for_message[message_id]].append(message_id)
        for part, (parent_id, ids) in enumerate(by_parent.items(), start=1):
            child_id = f"v5-child-{event_id}-{part}"
            add_child(parent_id, topic, str(event["answer"]), ids, child_id)

    # Add distractor children from non-event chatter, grouped by parent.
    fillers_by_parent_topic: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in corpus:
        message_id = str(row["message_id"])
        topic = filler_topic_by_id.get(message_id)
        if topic is not None:
            fillers_by_parent_topic[(parent_for_message[message_id], topic)].append(message_id)
    for index, ((parent_id, topic), ids) in enumerate(fillers_by_parent_topic.items(), start=1):
        add_child(parent_id, topic, "群友围绕游戏、动画和日常资讯的零散交流。", ids, f"v5-child-filler-{index:04d}")

    queries: list[dict[str, object]] = []
    for event in events:
        query = event.get("query")
        if not query:
            continue
        event_id = str(event["event_id"])
        answer_message_id = event_answer_message_id[event_id]
        query["gold_evidence_groups"] = [{
            "group_id": f"{query['qid']}_e1",
            "event_id": event_id,
            "message_ids": [answer_message_id],
            "child_id": child_for_message[answer_message_id],
        }]
        query["expected_evidence_blocks"] = 1
        query["room_id"] = ROOM_ID
        queries.append(query)

    # 100 no-time questions: 50 single-evidence, 35 two-evidence, 15 three-evidence.
    for index in range(100):
        evidence_count = 1 if index < 50 else (2 if index < 85 else 3)
        start = (index * 7) % len(plain_events)
        selected = [plain_events[(start + offset * 37) % len(plain_events)] for offset in range(evidence_count)]
        parts = [str(event["query_text"]) for event in selected]
        question = parts[0] if len(parts) == 1 else "；另外，".join(parts)
        gold_groups = []
        for event in selected:
            event_id = str(event["event_id"])
            answer_message_id = event_answer_message_id[event_id]
            gold_groups.append({
                "group_id": f"plain_{index + 1:03d}_{len(gold_groups) + 1}",
                "event_id": event_id,
                "message_ids": [answer_message_id],
                "child_id": child_for_message[answer_message_id],
            })
        queries.append({
            "qid": f"plain_{index + 1:03d}",
            "question": question,
            "answerable": True,
            "has_time_intent": False,
            "kind": "plain",
            "gold_evidence_groups": gold_groups,
            "expected_evidence_blocks": len(gold_groups),
            "room_id": ROOM_ID,
        })

    if len(corpus) != MESSAGE_COUNT or len(queries) != 200 or len(PEOPLE) != 50:
        raise ValueError((len(corpus), len(queries), len(PEOPLE)))

    # Ensure every gold answer is represented by an indexed child.
    child_ids = {str(child["id"]) for child in child_chunks}
    for query in queries:
        for evidence in query["gold_evidence_groups"]:
            if evidence["child_id"] not in child_ids:
                raise ValueError(f"missing gold child: {evidence['child_id']}")

    return corpus, queries, child_chunks, parents


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    corpus, queries, children, parents = build()
    write_jsonl(OUT_DIR / "corpus.jsonl", corpus)
    write_jsonl(OUT_DIR / "queries.jsonl", queries)
    write_jsonl(OUT_DIR / "child_chunks.jsonl", children)
    write_jsonl(OUT_DIR / "parent_chunks.jsonl", parents)
    print(
        f"Generated {len(corpus)} messages, {len(PEOPLE)} senders, "
        f"{len(parents)} parents, {len(children)} children, {len(queries)} questions."
    )
    print("Questions: 100 temporal, 100 without time; no-answer generation is disabled.")


if __name__ == "__main__":
    main()
