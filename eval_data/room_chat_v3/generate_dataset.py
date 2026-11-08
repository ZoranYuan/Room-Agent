"""Generate a deterministic Chinese group-chat retrieval benchmark."""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path


OUT_DIR = Path(__file__).parent
ROOM_ID = "eval-room-v3"
CONVERSATION_ID = "eval-conversation-v3"
SEED = 20261103
MESSAGE_COUNT = 3000

SURNAMES = [
    "林", "陈", "周", "吴", "郑", "王", "冯", "蒋", "沈", "韩",
    "杨", "朱", "秦", "许", "何", "吕", "施", "张", "孔", "曹",
]
GIVEN_NAMES = ["晓安", "知夏", "亦辰", "嘉宁", "星野"]
PEOPLE = [
    (f"u{index:03d}", surname + given)
    for index, (surname, given) in enumerate(
        ((surname, given) for surname in SURNAMES for given in GIVEN_NAMES),
        start=1,
    )
]

# Single-block questions: one short topical exchange contains the answer.
SINGLE_CASES = [
    ("葬送的芙莉莲", "最近想补一部节奏舒服的番。", "我推荐《葬送的芙莉莲》，旅行感和人物对白都很耐看。", "这个我也存到片单里了。", "群里推荐补哪部节奏比较舒服的番？", "《葬送的芙莉莲》"),
    ("迷宫饭", "这两天大家在聊奇幻题材。", "《迷宫饭》把迷宫冒险和料理放在一起，这个设定挺有意思。", "听起来不像普通打怪番。", "哪部作品把迷宫冒险和料理结合起来？", "《迷宫饭》"),
    ("间谍过家家", "想找一部能和家里人一起看的动画。", "《间谍过家家》比较合适，日常喜剧多，气氛也轻松。", "那我周末开给他们看。", "群里觉得哪部动画适合和家人一起看？", "《间谍过家家》"),
    ("排球少年", "最近又有人开始重看运动番了。", "我最喜欢《排球少年》里队友慢慢磨合、把配合练出来的部分。", "这种成长线每次看都热血。", "有人重看《排球少年》时最喜欢哪类情节？", "队友磨合并练出配合的成长情节"),
    ("灌篮高手", "老动画的片头曲真的很容易让人想起学生时代。", "《灌篮高手》的片头一响，我就会想到放学后守着电视看的时候。", "突然有画面了。", "哪部老动画的片头让群友想起学生时代？", "《灌篮高手》"),
    ("新世纪福音战士", "最近想找一部看完还能聊很久的老作品。", "《新世纪福音战士》吧，人物和结局都很适合看完再讨论。", "这部确实有好多不同解读。", "哪部老作品被推荐用来一起讨论人物和结局？", "《新世纪福音战士》"),
    ("药屋少女的呢喃", "有人问最近哪部动画的女主角比较有记忆点。", "《药屋少女的呢喃》的猫猫很鲜活，观察力强，说话也有自己的风格。", "她拆谜题时特别有意思。", "哪部作品的猫猫被夸观察力强、很有个性？", "《药屋少女的呢喃》"),
    ("星际牛仔", "想找一部音乐和画面都很有辨识度的旧番。", "《星际牛仔》的配乐很突出，爵士和蓝调的感觉一下就出来了。", "我先去找原声听听。", "哪部旧番因为爵士和蓝调风格的配乐被提到？", "《星际牛仔》"),
    ("侧耳倾听", "周末想看一部不太吵、看完心情会变好的动画电影。", "可以看《侧耳倾听》，它讲成长和喜欢做的事，氛围很温柔。", "这个适合慢慢看。", "群里推荐的温柔成长向动画电影是哪部？", "《侧耳倾听》"),
    ("海贼王", "长篇动画里我还是更喜欢一路冒险的感觉。", "《海贼王》最吸引我的就是伙伴一起出海、每一站都有新故事。", "世界观确实越铺越大。", "哪部长篇动画被形容为伙伴一路出海冒险？", "《海贼王》"),
    ("名侦探柯南", "小时候总猜不到案件最后是谁做的。", "《名侦探柯南》最有意思的还是线索藏得细，揭晓前可以自己推一遍。", "我现在看也会暂停猜凶手。", "群友觉得哪部作品适合边看边推理案件？", "《名侦探柯南》"),
    ("蓝色时期", "有没有讲创作、又不只是讲天赋的作品？", "《蓝色时期》挺合适，里面更多是在讲练习、选择和怎么找到自己的表达。", "这个角度我喜欢。", "哪部作品被推荐为关于创作练习和自我表达？", "《蓝色时期》"),
    ("电锯人", "最近有人聊到动画改编的镜头语言。", "《电锯人》的镜头处理挺像电影，氛围和停顿都很有辨识度。", "我记得片尾曲也每集不一样。", "哪部动画被说镜头语言很有电影感？", "《电锯人》"),
    ("火影忍者", "少年漫里哪段比赛最让人印象深？", "我会选《火影忍者》的中忍考试，角色多，比赛里也能看出各自的成长。", "那一段对战确实很密集。", "群里提到《火影忍者》的哪段比赛？", "中忍考试"),
    ("JOJO的奇妙冒险", "朋友说想看画风特别一点的动画。", "可以试试《JOJO的奇妙冒险》，夸张的姿势和色彩一眼就认得出来。", "这个推荐理由很有说服力。", "哪部动画因姿势和色彩夸张而被推荐？", "《JOJO的奇妙冒险》"),
    ("灵能百分百", "有没有主角很强但重点不是打赢所有人的动画？", "《灵能百分百》不错，重点其实是灵幻和龙套怎么理解自己、和别人相处。", "听起来比单纯升级有意思。", "哪部动画被推荐为重点在自我成长和人际关系？", "《灵能百分百》"),
    ("文豪野犬", "我想找一部异能设定、又有群像角色的作品。", "《文豪野犬》可以试试，角色阵营多，异能也和人物特点绑得比较紧。", "那我从第一季开始。", "哪部作品被推荐为异能群像题材？", "《文豪野犬》"),
    ("孤独摇滚！", "最近想看点音乐题材的轻松番。", "《孤独摇滚！》挺好笑的，社恐主角上台前后的反差特别明显。", "我看过片段，节奏很快。", "哪部音乐番因为社恐主角的反差被推荐？", "《孤独摇滚！》"),
    ("轻音少女", "想找部适合吃饭时放着看的日常动画。", "《轻音少女》就很舒服，轻音部练习不多，大家聊天和吃点心的戏很多。", "这个听起来完全不赶剧情。", "哪部日常动画被说有很多社团聊天和吃点心的片段？", "《轻音少女》"),
    ("紫罗兰永恒花园", "最近想看画面细腻一点、情绪比较克制的作品。", "《紫罗兰永恒花园》可以，信件和人物情感那条线很细腻。", "我准备好纸巾再看。", "哪部作品因信件和细腻情感线被推荐？", "《紫罗兰永恒花园》"),
    ("进击的巨人", "以前追番时最怕每集结尾留悬念。", "《进击的巨人》就是这种，一集结束总会冒出更多问题。", "所以很容易接着点下一集。", "哪部动画被形容为每集结尾都留下很多悬念？", "《进击的巨人》"),
    ("钢之炼金术师FA", "有没有伏笔收得比较完整的长篇动画？", "我推荐《钢之炼金术师FA》，主线完整，前面不少细节后面都能对上。", "好，我记下了。", "哪部作品被推荐为主线完整、伏笔能回收？", "《钢之炼金术师FA》"),
    ("虫师", "今晚不想看打斗，想看一集一集独立的小故事。", "《虫师》很适合，节奏安静，每一集像一个带点奇幻色彩的民间故事。", "我先从第一集试试。", "哪部动画被推荐为节奏安静、单集故事独立？", "《虫师》"),
    ("奇巧计程车", "有人想看一部线索能互相串起来的短篇动画。", "《奇巧计程车》可以，前面看似不相关的对话后面会慢慢连起来。", "那我得认真听台词。", "哪部短篇动画被说前后对话线索会串起来？", "《奇巧计程车》"),
    ("黄金神威", "最近又想看有历史背景、角色还挺有趣的番。", "《黄金神威》吧，冒险线之外还有很多饮食和地方文化细节。", "看起来题材挺特别。", "哪部作品被提到兼有冒险、饮食和地方文化？", "《黄金神威》"),
    ("乐队的夏天", "最近想回顾一下中文乐队现场。", "我会翻《乐队的夏天》，不同乐队的编曲风格差异特别明显。", "我记得里面有几首歌后来循环了很久。", "群里推荐回看哪个乐队节目？", "《乐队的夏天》"),
    ("喜剧之王", "想找一部老电影，既好笑又有点心酸。", "周星驰的《喜剧之王》吧，很多笑点背后其实是小人物想被认可。", "这部我很久没重看了。", "哪部周星驰电影被形容为笑中带着小人物的辛酸？", "《喜剧之王》"),
    ("塞尔达传说：旷野之息", "最近想玩一款能随便探索、不太催任务的游戏。", "《塞尔达传说：旷野之息》很适合，地图里经常走着走着就发现新东西。", "我最喜欢爬到高处看地形。", "哪款游戏被推荐为可以自由探索、常有意外发现？", "《塞尔达传说：旷野之息》"),
    ("坂本龙一", "最近整理电影原声，想找一张适合安静听的专辑。", "坂本龙一的作品很适合，钢琴和留白都很克制。", "我晚上戴耳机试试。", "群里推荐了哪位音乐人的作品来安静听？", "坂本龙一"),
    ("名侦探柯南剧场版", "假期想挑一部节奏快、适合大家一起看的动画电影。", "可以选《名侦探柯南：贝克街的亡灵》，推理和虚拟世界设定都挺抓人。", "我小时候看过，想再补一次。", "推荐一起重看的柯南剧场版是哪一部？", "《名侦探柯南：贝克街的亡灵》"),
]

# Multi-block questions: each fact is placed in a distant part of the room history.
# Every multi-block question requires exactly three distinct evidence blocks.
MULTI_CASES = [
    ("周末补番", "我投《葬送的芙莉莲》，想看它的旅行和日常。", "周六晚上八点开投影吧。", "先看两集，看到十点就收。", "大家选了哪部番、几点开始、准备看几集？", "《葬送的芙莉莲》；周六晚八点；两集", 3),
    ("老番重温", "我想重看《灌篮高手》，比赛部分现在看还是很燃。", "那周日下午三点在客厅集合。", "我们只看全国大赛那几集，不从头补。", "重看的作品、集合时间和观看范围是什么？", "《灌篮高手》；周日下午三点；全国大赛相关剧集", 3),
    ("电影夜", "动画电影我选《千与千寻》，大家都比较熟。", "周五九点线上开播，链接我提前发。", "看到一半可以暂停十分钟，方便大家拿吃的。", "电影夜看哪部、几点开始、中途怎么安排？", "《千与千寻》；周五晚九点；中途暂停十分钟", 3),
    ("乐队现场", "如果去音乐节，我最想听草东没有派对。", "票买周日场，下午四点在东门碰头。", "结束后不赶场，直接坐地铁回去。", "最想听的乐队、哪天几点碰面、散场后怎么走？", "草东没有派对；周日下午四点东门；结束后坐地铁回去", 3),
    ("桌游聚会", "我带《璀璨宝石》，规则不复杂，新手也能玩。", "周六两点去店里，先坐靠窗那桌。", "第一局不计分，大家熟悉规则再正式玩。", "桌游选哪款、几点去、第一局怎么安排？", "《璀璨宝石》；周六下午两点；第一局不计分熟悉规则", 3),
    ("漫画交换", "我想带《蓝色时期》去交换，最近刚好读完一卷。", "周三下班后在地铁站旁咖啡馆见。", "每人带一本就行，交换前先聊五分钟读后感。", "交换时带哪部漫画、在哪里见、怎么开始？", "《蓝色时期》；地铁站旁咖啡馆；先聊五分钟读后感", 3),
    ("动画配乐", "我想分享《星际牛仔》的原声，爵士那几首很耐听。", "周日晚七点开语音房，我把歌单放共享文档。", "先听前三首，再聊各自最喜欢哪段配乐。", "分享什么原声、几点开房、先听几首？", "《星际牛仔》原声；周日晚七点；前三首", 3),
    ("影展报名", "我报了《机器人之梦》，想看完和大家聊友情主题。", "影厅选周六下午一点半那场。", "散场后在一楼长椅等十分钟再去吃饭。", "报名了哪部电影、几点场、散场后在哪等？", "《机器人之梦》；周六下午一点半；一楼长椅等十分钟", 3),
    ("游戏联机", "这周联机玩《胡闹厨房2》吧，合作起来比较热闹。", "周四晚上九点进语音频道，我建房间。", "每局换一次队长，免得一个人一直指挥。", "联机玩什么、几点进频道、队伍怎么轮换？", "《胡闹厨房2》；周四晚九点；每局轮换队长", 3),
    ("综艺回看", "我想回看《乐队的夏天》里那场改编舞台。", "周二晚八点我发视频链接，大家线上看。", "看完各自选一首最喜欢的，不投票排名。", "回看哪个节目、什么时候看、看完做什么？", "《乐队的夏天》；周二晚八点；各选一首喜欢的歌", 3),
    ("国漫讨论", "最近补《罗小黑战记》，动作设计比我想的还细。", "周日下午两点在群里开讨论语音。", "先聊电影版，再聊动画剧集，不剧透后续。", "补了哪部作品、几点讨论、按什么顺序聊？", "《罗小黑战记》；周日下午两点；先电影版再动画剧集", 3),
    ("科幻电影", "我推荐《流浪地球2》，设定细节挺适合暂停讨论。", "周五晚上七点半到小周家看。", "大家带自己的杯子，饮料统一点无糖的。", "推荐哪部电影、几点在哪里看、饮料怎么选？", "《流浪地球2》；周五晚七点半小周家；点无糖饮料", 3),
    ("声优作品", "我想听《紫罗兰永恒花园》的配乐，情绪推进做得很好。", "周日早上十点在共享歌单里一起听。", "每个人挑一段配乐，说说对应哪封信。", "一起听哪部作品的配乐、几点、每人分享什么？", "《紫罗兰永恒花园》配乐；周日上午十点；分享配乐对应的信件", 3),
    ("复古动画", "我想重看《魔女宅急便》，城市和飞行的画面很放松。", "约周六下午两点在线上同步播放。", "字幕用普通话版，避免版本不同对不上。", "重看哪部电影、什么时候看、选什么字幕？", "《魔女宅急便》；周六下午两点；普通话字幕版", 3),
    ("推理剧", "最近想补《非自然死亡》，每集案件都能聊几句。", "周一到周五每天看一集，晚上九点发起。", "讨论时不提前搜结局，先各自猜一次。", "补哪部剧、每天几点看几集、讨论时有什么约定？", "《非自然死亡》；每晚九点一集；不提前搜结局先猜", 3),
    ("声优见面会", "有人转了花泽香菜的访谈，我觉得她聊配音准备很认真。", "周六下午三点看回放，我把链接存群公告。", "只看前半小时，后面留给大家聊印象。", "看谁的访谈、几点看、看多久？", "花泽香菜访谈；周六下午三点；前半小时", 3),
    ("漫画改编", "《迷宫饭》动画我觉得料理部分还原得挺有趣。", "周三晚上八点一起看最新一集。", "看完不急着散，留十五分钟聊改编差异。", "看哪部作品、几点看、看完留多久讨论？", "《迷宫饭》；周三晚八点；讨论十五分钟", 3),
    ("游戏原声", "我最近循环《女神异闻录5》的原声，城市感很强。", "周五下班后去唱片店试听。", "预算最多两百，先听二手盘有没有划痕。", "想找哪款游戏的原声、什么时候去、预算多少？", "《女神异闻录5》原声；周五下班后；两百元以内", 3),
    ("经典电影", "我投《天堂电影院》，想在大屏上看一次。", "周六晚七点去社区影厅，票我先订。", "散场后不安排聚餐，大家赶末班车。", "选了哪部电影、几点在哪里看、散场后怎么安排？", "《天堂电影院》；周六晚七点社区影厅；散场后各自回家", 3),
    ("配音练习", "我想用《千与千寻》练一小段配音，台词比较熟。", "周四晚上八点线上练，每人录一分钟。", "文件发群里就好，不做公开发布。", "用什么作品练配音、何时练、每人录多久？", "《千与千寻》；周四晚八点；每人一分钟", 3),
    ("动画投票", "我选《强风吹拂》，想看团队从不熟到配合起来。", "周日晚上七点半开始同步看。", "第一晚只看三集，后面再决定要不要继续。", "选了哪部动画、几点开始、第一晚看几集？", "《强风吹拂》；周日晚七点半；三集", 3),
    ("旧剧重看", "我想重看《甄嬛传》，台词现在听还是很有味道。", "那我们周六下午两点开着看。", "先看华妃出场那几集，不从第一集开始。", "重看的剧、开始时间和先看哪段？", "《甄嬛传》；周六下午两点；华妃出场部分", 3),
    ("电影选择", "最近想看《银翼杀手2049》，画面和配乐都很适合大屏。", "周五晚上九点我家投影，地址私聊发。", "字幕选简体中文字幕版，免得大家对不上台词。", "选了哪部电影、几点看、字幕用哪个版本？", "《银翼杀手2049》；周五晚九点；简体中文字幕版", 3),
    ("音乐分享", "我最近听陈粒的现场专辑，编曲比录音室版本更松弛。", "周日下午我把链接贴到群里。", "我会放到网易云歌单里，设成群里都能听。", "分享谁的哪种专辑、什么时候发、放在哪个平台？", "陈粒现场专辑；周日下午；网易云歌单", 3),
    ("经典动画", "《攻壳机动队》剧场版值得重看，老动画的画面细节很耐看。", "周三晚上八点线上放映。", "这次用日文原声中文字幕版，别开配音版。", "重看哪部作品、几点放映、选什么音轨版本？", "《攻壳机动队》剧场版；周三晚八点；日文原声中文字幕版", 3),
    ("舞台剧", "我想看《暗恋桃花源》，这次巡演据说舞美改过。", "周日场还有票，我先确认座位再回群里。", "我倾向买中间偏后的位置，视野比较完整。", "想看哪部舞台剧、哪天的场次、倾向什么座位？", "《暗恋桃花源》；周日场；中间偏后座位", 3),
    ("音乐剧", "《摇滚莫扎特》的现场片段挺有感染力，想找全场录制看看。", "周二晚上我把正版平台链接发出来。", "先看前半场，重点听乐队编曲，不急着一次看完。", "想看哪部音乐剧、什么时候发链接、先看哪部分？", "《摇滚莫扎特》；周二晚上；先看前半场", 3),
    ("动画电影", "我想看《红辣椒》，梦境和现实交错那段很特别。", "周六下午四点同步播放，结束后聊十分钟。", "字幕用简体中文，片尾曲放完再结束语音。", "看哪部动画电影、几点开始、字幕和结束安排是什么？", "《红辣椒》；周六下午四点；简体中文字幕，片尾后结束", 3),
    ("游戏聚会", "这周想玩《双人成行》，关卡设计挺适合两个人合作。", "周五晚上八点开机，手柄我来带。", "先从第一章开始，暂时不看攻略。", "玩什么游戏、几点开始、攻略怎么处理？", "《双人成行》；周五晚八点；从第一章开始且不看攻略", 3),
    ("动画配音", "我想找《大鱼海棠》的配音片段，配乐和画面都挺有记忆点。", "周日晚七点我把片段链接发群里。", "只截椿第一次见到鲲的那段，长度控制在两分钟。", "分享哪部作品的配音片段、几点发、截哪段？", "《大鱼海棠》；周日晚七点；椿第一次见到鲲的片段", 3),
]

STOCK_CASES = [
    (
        "沪深300",
        (
            "聊到沪深300时，有人说指数回到3800点附近，接下来先看成交量和估值，不急着追高。",
            "聊到沪深300时，有人说成交量放大但指数还没站稳，先观察几天，不急着加仓。",
            "最近聊到沪深300时，有人说指数站回关键区间但短期涨幅不小，还是不建议追高。",
        ),
        (
            "先看成交量和估值，不急着追高。",
            "成交量放大但指数未站稳，先观察、不急着加仓。",
            "指数站回关键区间但短期涨幅不小，仍不建议追高。",
        ),
    ),
    (
        "宁德时代",
        (
            "聊到宁德时代时，有人建议先看财报里的毛利率和订单，不按单日股价涨跌做决定。",
            "聊到宁德时代时，有人提到订单还要观察兑现情况，先看季报，不把短线反弹当成趋势。",
            "最近聊到宁德时代时，有人觉得海外订单和毛利率仍是观察重点，暂时不追涨。",
        ),
        (
            "重点看财报毛利率和订单，不按单日涨跌做决定。",
            "观察订单兑现和季报，不把短线反弹当趋势。",
            "继续关注海外订单与毛利率，暂时不追涨。",
        ),
    ),
    (
        "贵州茅台",
        (
            "聊到贵州茅台时，有人建议先关注渠道库存和财报，不用短期价格变化猜长期价值。",
            "聊到贵州茅台时，有人提到渠道库存变化还不明朗，先等后续数据再讨论。",
            "最近聊到贵州茅台时，有人说渠道库存和财报仍要继续观察，不因一两天波动下结论。",
        ),
        (
            "关注渠道库存和财报，不用短期价格变化猜长期价值。",
            "渠道库存变化不明朗，等后续数据再讨论。",
            "继续观察渠道库存和财报，不因短期波动下结论。",
        ),
    ),
    (
        "黄金ETF",
        (
            "聊到黄金ETF时，有人说要同时看金价和利率变化，不把短期上涨当成确定趋势。",
            "聊到黄金ETF时，有人提到利率预期变化较大，先观察回撤，不急着追涨。",
            "最近聊到黄金ETF时，有人认为金价波动还在，继续观察利率和回撤情况。",
        ),
        (
            "同时看金价和利率，不把短期上涨当确定趋势。",
            "利率预期变化较大，先观察回撤、不急着追涨。",
            "继续观察利率和回撤情况。",
        ),
    ),
]


FILLER_DATA = [
    ("追番", ["下一集", "片尾", "字幕", "更新"], ["留到周末", "看到十一点就停", "先把弹幕关掉", "补完再看评论"]),
    ("听歌", ["通勤歌单", "现场录音", "循环播放", "耳机音量"], ["加一首慢歌", "存到共享歌单", "换个版本听听", "别开太大声"]),
    ("打游戏", ["每日任务", "新地图", "组队副本", "手柄电量"], ["先清日常", "等人齐再开", "周末再研究", "记得带充电线"]),
    ("吃东西", ["夜宵", "冰箱里的菜", "新开的面馆", "下午茶"], ["少点一点", "下次一起去", "先问清楚辣度", "打包留一份"]),
    ("通勤", ["地铁换乘", "共享单车", "雨伞", "下班路线"], ["提前一站下车", "错开晚高峰", "把伞放包里", "走人少的那条路"]),
    ("看书", ["书签", "读书进度", "电子书", "书架"], ["每天读十页", "先看目录", "周末再整理", "把喜欢的段落记下来"]),
    ("拍照", ["手机相册", "夜景", "镜头", "照片备份"], ["先删重复的", "等天暗一点", "拍完就备份", "别忘了擦镜头"]),
    ("运动", ["散步", "拉伸", "跑步鞋", "步数"], ["先慢慢来", "下雨就改室内", "记得热身", "别一上来加量"]),
    ("做饭", ["调味", "剩菜", "电饭煲", "洗碗"], ["先少放盐", "明天带饭", "用完记得拔电", "轮流收拾"]),
    ("日常", ["桌面", "充电器", "日历提醒", "快递"], ["顺手收一下", "放到固定位置", "设个提醒", "下班路上取"]),
]
FILLER_STARTS = [
    "刚刚弄{subject}的时候发现，{detail}。",
    "最近我在留意{subject}，主要是{detail}。",
    "关于{subject}，我最近发现{detail}。",
    "你们处理{subject}时会不会也{detail}？",
    "我想把{subject}这件事调整一下，{detail}。",
]
FILLER_REPLIES = [
    "对，处理{subject}时我也踩过坑，后来就{tip}。",
    "哈哈确实，处理{subject}时{tip}会省心一点。",
    "我之前没想到{subject}这点，看来得{tip}。",
    "这个建议可以，处理{subject}时我晚点试试{tip}。",
    "有道理，{subject}这事我先照着{tip}做。",
]
FILLER_FOLLOWUPS = [
    "我一般弄{subject}时会{tip}，不然很容易忘。",
    "下次碰到{subject}，要不就{tip}？",
    "说起来我上次处理{subject}也是{tip}，效果还行。",
    "那我先处理一下{subject}，具体就{tip}。",
    "行，{subject}我记下了，回头试试{tip}。",
]
FILLER_CLOSERS = [
    "好，那我按这个来，后面再同步。", "收到，我也试试这个办法。",
    "哈哈先这样，等我忙完再接着聊。", "可以，我弄好以后发群里。",
    "懂了，这下知道怎么处理了。", "行，我晚点确认一下再说。",
    "记下了，免得下次又临时找。", "好主意，我正好也需要整理一下。",
    "那就先这样，遇到问题再一起想。", "我先去处理，晚点回来补消息。",
    "谢谢提醒，差点就忘了这件事。", "收到，我把这个加进待办里。",
    "嗯嗯，这样安排起来清楚多了。", "可以的，我看看手边有没有合适的。",
    "好，等我试过之后说下感受。", "这个点很实用，先收藏了。",
    "哈哈我之前也踩过这个坑。", "行，那我就不另外折腾了。",
    "好的，今晚有空我就弄一下。", "有道理，先从最简单的开始。",
    "收到，明天我再看看具体情况。", "可以，之后有变化群里说一声。",
    "懂了，我刚刚还在纠结这个。", "好，那先不展开了，回头再聊。",
    "感谢分享，我之前真没想到。", "嗯，这个办法比我原来想的省事。",
    "我先去忙啦，大家晚点见。", "记好了，之后就照这个习惯来。",
    "行，我把相关东西先放到一起。", "哈哈聊着聊着又记下一件事。",
]
FILLER_BRIDGES = {
    "追番": ["最近追番的人好像都在补旧作。", "新一季开播后片单又变长了。", "我现在尽量不同时开太多部。", "等周末有整块时间再认真看。", "追到一半停几天就容易忘剧情。", "看完一集先不急着点下一集。"],
    "听歌": ["最近大家分享的现场版本还挺多。", "有些歌换个耳机听感觉差不少。", "我打算周末把收藏的歌单理一遍。", "新歌先听几遍再决定要不要收藏。", "通勤路上听歌时间过得快一点。", "最近循环的几首歌风格差得挺大。"],
    "打游戏": ["版本更新以后菜单位置都变了。", "我得先把设置调回习惯的样子。", "最近玩游戏还是想找点轻松的。", "人齐的时候开一局确实更有意思。", "我准备先把教程和操作过一遍。", "这两天在线时间不太固定。"],
    "吃东西": ["附近新开的店最近排队有点长。", "天气凉下来以后更想喝热汤。", "我发现午饭吃太晚下午会没精神。", "下次点单可以多试一个没吃过的。", "冰箱里还有点食材得尽快用掉。", "最近外卖软件老给我推同几家。"],
    "通勤": ["早高峰那段路最近又开始堵了。", "我发现换一条街走会安静很多。", "下雨天还是坐地铁比较稳妥。", "最近晚点的时候比上个月多。", "我准备把常用路线存到地图里。", "骑车的话记得看一下电量。"],
    "看书": ["我看书慢主要是老想回头查资料。", "最近借的几本书还没来得及拆封。", "有些段落隔几天再看会有新感觉。", "周末图书馆人少一点比较安静。", "我准备把读完的书单单独记下来。", "最近更喜欢短章节的书。"],
    "拍照": ["光线变一点，照片的感觉就差很多。", "我现在拍完会先挑片再修图。", "相册空间不够得找时间清一清。", "最近手机夜景模式比以前稳了。", "我想试试同一个地方不同时间拍。", "有些照片当时没感觉，过几天再看还不错。"],
    "运动": ["最近坐久了确实该多起来活动。", "我把运动鞋放门口就更容易出门。", "天气变化大，运动前得先看看预报。", "拉伸做足了第二天会舒服不少。", "我现在更在意能不能长期坚持。", "运动完早点睡比继续刷手机强。"],
    "做饭": ["最近想少点外卖，自己做点简单的。", "有些菜看着步骤多其实挺快。", "我准备把常用调料放到顺手的位置。", "洗菜的时候顺便把台面收一下。", "下次可以多做一点留着第二天吃。", "做饭最难的是决定今天吃什么。"],
    "日常": ["小事记下来以后脑子轻松很多。", "最近待办有点多，得按优先级排。", "我准备把桌面上不常用的东西收起来。", "东西固定放一个位置真的省时间。", "这周过得好快，转眼又到周末了。", "最近天气舒服，出门心情都好点。"],
}

NO_ANSWER_TITLES = [
    "《来自深渊》", "《昭和元禄落语心中》", "《比宇宙更远的地方》",
    "《夏目友人帐》", "《白箱》", "《少女终末旅行》", "《异度侵入》",
    "《四叠半神话大系》", "《花开伊吕波》", "《冰菓》", "《日常》",
    "《昭和元禄落语心中》剧场版", "《平家物语》", "《乒乓》", "《妄想代理人》",
]


class DatasetBuilder:
    def __init__(self) -> None:
        self.rng = random.Random(SEED)
        self.messages: list[dict[str, object]] = []
        self.queries: list[dict[str, object]] = []
        self.elapsed = 0

    def add_message(self, content: str, sender_index: int | None = None) -> str:
        sender_id, sender_name = (
            self.rng.choice(PEOPLE)
            if sender_index is None
            else PEOPLE[sender_index % len(PEOPLE)]
        )
        self.elapsed += self.rng.randint(20, 1800)
        message_id = f"v3-m{len(self.messages) + 1:06d}"
        base_time = datetime(2025, 9, 1, tzinfo=timezone.utc)
        send_time = int((base_time + timedelta(seconds=self.elapsed)).timestamp())
        self.messages.append(
            {
                "message_id": message_id,
                "room_id": ROOM_ID,
                "conversation_id": CONVERSATION_ID,
                "sender_id": sender_id,
                "sender_name": sender_name,
                "seq": len(self.messages) + 1,
                "type": 0,
                "content": content,
                "status": 0,
                "send_time": send_time,
            }
        )
        return message_id

    def add_filler_vignette(self) -> None:
        activity, subjects, tips = self.rng.choice(FILLER_DATA)
        subject = self.rng.choice(subjects)
        tip = self.rng.choice(tips)
        details = [
            "提前准备会省不少事", "不想每次都临时找", "有一步很容易漏掉",
            "最近总是拖到很晚", "上次差点就忘了",
        ]
        detail = self.rng.choice(details)
        lines = [
            self.rng.choice(FILLER_STARTS).format(subject=subject, detail=detail),
            self.rng.choice(FILLER_REPLIES).format(subject=subject, tip=tip),
            self.rng.choice(FILLER_FOLLOWUPS).format(subject=subject, tip=tip),
            f"{subject}这块我最近也在留意。 {self.rng.choice(FILLER_BRIDGES[activity])}",
            f"关于{subject}，{self.rng.choice(FILLER_CLOSERS)}",
        ]
        for line in lines:
            self.add_message(line)

    def add_query(
        self,
        *,
        kind: str,
        question: str,
        answer: str,
        evidence_groups: list[list[str]],
        category: str,
        time_hint: dict[str, str] | None = None,
    ) -> None:
        query = {
            "qid": f"{kind}_{sum(row['qid'].startswith(kind + '_') for row in self.queries) + 1:03d}",
            "room_id": ROOM_ID,
            "question": question,
            "answerable": bool(evidence_groups),
            "answer": answer,
            # Evidence groups map source messages to production child chunks.
            # Retrieval metrics count matched child chunks, not message seqs.
            "gold_evidence_groups": [
                {"group_id": f"{kind}_{len(self.queries) + 1:03d}_e{index + 1}", "message_ids": ids}
                for index, ids in enumerate(evidence_groups)
            ],
            "expected_evidence_blocks": len(evidence_groups),
            "category": category,
            "source": "synthetic_curated_v3",
        }
        if time_hint is not None:
            query["time_hint"] = time_hint
        self.queries.append(query)

    def add_single_case(self, case: tuple[str, str, str, str, str, str]) -> None:
        topic, lead, fact, reply, question, answer = case
        self.add_message(lead)
        anchor_id = self.add_message(fact)
        self.add_message(reply)
        self.add_query(
            kind="single",
            question=question,
            answer=answer,
            evidence_groups=[[anchor_id]],
            category=f"single_block:{topic}",
        )

    def add_multi_part(self, case: tuple[object, ...], part_index: int) -> str:
        topic = str(case[0])
        fact = str(case[1 + part_index])
        anchor_id = self.add_message(fact)
        response = ["这个我也记下了。", "好，信息对上了。", "收到，等确定后我跟上。"]
        self.add_message(self.rng.choice(response))
        self.add_message(f"关于{topic}我先把这点记下来，免得后面和别的安排混了。")
        return anchor_id

    def add_stock_phase(self, phase_index: int) -> dict[str, str]:
        anchors: dict[str, str] = {}
        replies = (
            "这只是群里的观察，不当作确定的涨跌结论。",
            "先把这条记下来，等有新信息再接着讨论。",
        )
        for topic, phase_messages, _ in STOCK_CASES:
            anchors[topic] = self.add_message(phase_messages[phase_index])
            self.add_message(self.rng.choice(replies))
        return anchors

    def add_temporal_stock_queries(
        self,
        middle_anchors: dict[str, str],
        late_anchors: dict[str, str],
    ) -> None:
        messages_by_id = {
            str(message["message_id"]): message for message in self.messages
        }
        local_timezone = timezone(timedelta(hours=8))

        for topic, _, answers in STOCK_CASES:
            middle_message = messages_by_id[middle_anchors[topic]]
            middle_time = datetime.fromtimestamp(
                int(middle_message["send_time"]), timezone.utc
            ).astimezone(local_timezone)
            day_start = middle_time.replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = middle_time.replace(hour=23, minute=59, second=59, microsecond=0)
            date_text = f"{middle_time.year}年{middle_time.month}月{middle_time.day}日"

            self.add_query(
                kind="temporal",
                question=f"{date_text}群里讨论{topic}时的判断是什么？",
                answer=answers[1],
                evidence_groups=[[middle_anchors[topic]]],
                category=f"temporal:range:{topic}",
                time_hint={
                    "mode": "range",
                    "start_at": day_start.isoformat(),
                    "end_at": day_end.isoformat(),
                },
            )
            self.add_query(
                kind="temporal",
                question=f"最近一次群里讨论{topic}时，大家的判断是什么？",
                answer=answers[2],
                evidence_groups=[[late_anchors[topic]]],
                category=f"temporal:latest:{topic}",
                time_hint={"mode": "latest"},
            )

    def write(self) -> None:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        for filename, rows in (("corpus.jsonl", self.messages), ("queries.jsonl", self.queries)):
            with (OUT_DIR / filename).open("w", encoding="utf-8") as file:
                for row in rows:
                    file.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_dataset() -> DatasetBuilder:
    builder = DatasetBuilder()

    # Early history: single-block evidence and the first part of every multi-block answer.
    for case in SINGLE_CASES:
        builder.add_single_case(case)
    builder.add_stock_phase(0)
    early_evidence: dict[str, list[str]] = {}
    for case in MULTI_CASES:
        early_evidence[str(case[0])] = [builder.add_multi_part(case, 0)]
    for _ in range(157):
        builder.add_filler_vignette()

    # Middle history: second evidence for three-block questions, surrounded by chatter.
    middle_stock_anchors = builder.add_stock_phase(1)
    middle_evidence: dict[str, list[str]] = {}
    for case in MULTI_CASES:
        middle_evidence[str(case[0])] = [builder.add_multi_part(case, 1)]
    for _ in range(186):
        builder.add_filler_vignette()

    # Late history: the third required evidence block for every multi-block question.
    late_stock_anchors = builder.add_stock_phase(2)
    late_evidence: dict[str, list[str]] = {}
    for case in MULTI_CASES:
        late_evidence[str(case[0])] = [builder.add_multi_part(case, 2)]
    for _ in range(180):
        builder.add_filler_vignette()

    for case in MULTI_CASES:
        topic = str(case[0])
        question, answer, expected_blocks = str(case[-3]), str(case[-2]), int(case[-1])
        groups = list(early_evidence[topic])
        groups.extend(middle_evidence[topic])
        groups.extend(late_evidence[topic])
        builder.add_query(
            kind="multi",
            question=question,
            answer=answer,
            evidence_groups=[[message_id] for message_id in groups],
            category=f"multi_block:{expected_blocks}:{topic}",
        )

    builder.add_temporal_stock_queries(
        middle_anchors=middle_stock_anchors,
        late_anchors=late_stock_anchors,
    )

    for index, title in enumerate(NO_ANSWER_TITLES, start=1):
        builder.add_query(
            kind="noanswer",
            question=f"群里最后决定哪天一起看{title}，在哪个平台开播？",
            answer="",
            evidence_groups=[],
            category="no_answer:unmentioned_title",
        )

    # Add deterministic harmless filler until the requested corpus size is reached.
    while len(builder.messages) + 5 <= MESSAGE_COUNT:
        builder.add_filler_vignette()
    while len(builder.messages) < MESSAGE_COUNT:
        builder.add_message("我先把前面提到的安排记下，后面有变化再在群里说。")

    # Ensure all 100 synthetic participants appear without changing message text or order.
    present = {str(row["sender_id"]) for row in builder.messages}
    missing = [person for person in PEOPLE if person[0] not in present]
    filler_rows = [row for row in builder.messages if str(row["message_id"]).startswith("v3-m")]
    for row, (sender_id, sender_name) in zip(filler_rows[-len(missing):] if missing else [], missing):
        row["sender_id"] = sender_id
        row["sender_name"] = sender_name

    if len(builder.messages) != MESSAGE_COUNT:
        raise ValueError(f"Expected {MESSAGE_COUNT} messages, got {len(builder.messages)}")
    if len({str(row["sender_id"]) for row in builder.messages}) != 100:
        raise ValueError("Expected all 100 participants to appear")
    if (sum(row["category"].startswith("single_block:") for row in builder.queries) != 30
            or sum(row["category"].startswith("multi_block:") for row in builder.queries) != 30
            or sum(row["category"].startswith("temporal:") for row in builder.queries) != 8
            or sum(not row["answerable"] for row in builder.queries) != 15):
        raise ValueError("Unexpected evaluation question distribution")

    all_message_ids = {str(row["message_id"]) for row in builder.messages}
    for query in builder.queries:
        for group in query["gold_evidence_groups"]:
            if not set(group["message_ids"]) <= all_message_ids:
                raise ValueError(f"Unknown gold evidence message in {query['qid']}")

    return builder


def main() -> None:
    builder = build_dataset()
    builder.write()
    print(
        f"Generated {len(builder.messages)} messages from "
        f"{len({row['sender_id'] for row in builder.messages})} participants; "
        f"questions={len(builder.queries)} (single=30, multi=30, temporal=8, no-answer=15)."
    )


if __name__ == "__main__":
    main()
