import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.font_manager import FontProperties
from PIL import Image, ImageDraw, ImageFont
from wordcloud import WordCloud

# 设置中文字体路径（根据系统调整）
font_candidates = [
    Path('C:/Windows/Fonts/simhei.ttf'),  # Windows
    Path('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'),  # Linux
    Path('/System/Library/Fonts/PingFang.ttc'),  # Mac
]
font_path = next((str(path) for path in font_candidates if path.exists()), None)
if font_path is None:
    raise FileNotFoundError("未找到可用的中文字体，请手动修改 font_path。")

title_font = FontProperties(fname=font_path)


def patch_textbbox_for_transposed_font():
    """兼容 Pillow 8.x 中 textbbox 不支持 TransposedFont 的问题。"""
    original_textbbox = ImageDraw.ImageDraw.textbbox

    def compat_textbbox(self, xy, text, font=None, anchor=None, spacing=4, align='left',
                        direction=None, features=None, language=None, stroke_width=0,
                        embedded_color=False):
        if isinstance(font, ImageFont.TransposedFont):
            x, y = xy
            width, height = font.getsize(
                text,
                direction=direction,
                features=features,
                language=language,
                stroke_width=stroke_width
            )
            return x, y, x + width, y + height
        return original_textbbox(
            self,
            xy,
            text,
            font=font,
            anchor=anchor,
            spacing=spacing,
            align=align,
            direction=direction,
            features=features,
            language=language,
            stroke_width=stroke_width,
            embedded_color=embedded_color
        )

    ImageDraw.ImageDraw.textbbox = compat_textbbox


patch_textbbox_for_transposed_font()

# 福州国潮香氛主题词汇数据（词语:权重）
words_freq = {
    # 核心主题（高权重）
    "福州香氛": 100,
    "国潮": 95,
    "嗅觉经济": 88,
    "本土文化": 85,
    "三坊七巷": 82,

    # 消费认知
    "茉莉花香": 78,
    "闽都文化": 75,
    "非遗制香": 72,
    "东方美学": 70,
    "文化自信": 68,
    "品质认知": 65,
    "品牌故事": 62,
    "寿山石韵": 58,
    "软木画艺": 55,

    # 消费行为
    "悦己消费": 60,
    "场景消费": 58,
    "礼品馈赠": 55,
    "空间香氛": 52,
    "茶道用香": 50,
    "书房雅集": 48,
    "个人护理": 45,
    "车载香氛": 42,
    "线下体验": 40,
    "线上种草": 38,

    # 购买意愿影响因素
    "价格敏感": 55,
    "包装设计": 52,
    "香调偏好": 50,
    "成分安全": 48,
    "品牌口碑": 46,
    "文化内涵": 45,
    "限量联名": 42,
    "KOL推荐": 40,
    "试用体验": 38,
    "复购意愿": 35,

    # 福州特色元素
    "脱胎漆器": 45,
    "油纸伞": 42,
    "榕城记忆": 40,
    "鼓山云雾": 38,
    "闽江晚风": 35,
    "鱼丸香": 32,
    "茉莉花茶": 48,
    "坊巷文化": 44,
    "闽南红砖": 36,
    "妈祖文化": 33,

    # 情感与趋势
    "情绪价值": 50,
    "仪式感": 45,
    "慢生活": 42,
    "治愈系": 40,
    "轻奢": 38,
    "Z世代": 36,
    "她经济": 35,
    "社交货币": 33,
    "收藏属性": 30,
    "国风": 42
}

# 生成文本（根据权重重复词语）
text = ""
for word, freq in words_freq.items():
    text += (word + " ") * int(freq / 3)  # 缩放权重以避免过大

# 创建香炉形状的 mask（800x800）
def create_incense_burner_mask():
    canvas_size = 800
    scale = 3
    mask = Image.new('L', (canvas_size * scale, canvas_size * scale), 0)
    draw = ImageDraw.Draw(mask)

    def s(value):
        return int(value * scale)

    # 炉盖：宝珠钮 + 穹顶盖
    draw.ellipse([s(365), s(86), s(435), s(152)], fill=255)
    draw.rounded_rectangle([s(386), s(144), s(414), s(186)], radius=s(10), fill=255)
    draw.ellipse([s(250), s(132), s(550), s(346)], fill=255)
    draw.rectangle([0, s(272), s(canvas_size), s(canvas_size)], fill=0)
    draw.rounded_rectangle([s(288), s(248), s(512), s(304)], radius=s(20), fill=255)
    draw.rounded_rectangle([s(332), s(214), s(468), s(248)], radius=s(12), fill=255)

    # 炉身：鼓腹炉体 + 口沿
    body_points = [
        (s(268), s(316)),
        (s(228), s(362)),
        (s(204), s(430)),
        (s(216), s(510)),
        (s(250), s(566)),
        (s(550), s(566)),
        (s(584), s(510)),
        (s(596), s(430)),
        (s(572), s(362)),
        (s(532), s(316)),
    ]
    draw.polygon(body_points, fill=255)
    draw.ellipse([s(190), s(332), s(610), s(584)], fill=255)
    draw.rounded_rectangle([s(238), s(294), s(562), s(348)], radius=s(18), fill=255)
    draw.rounded_rectangle([s(268), s(534), s(532), s(576)], radius=s(16), fill=255)

    # 双耳：环耳中间留白，让香炉特征更明显
    draw.ellipse([s(112), s(348), s(236), s(476)], fill=255)
    draw.ellipse([s(564), s(348), s(688), s(476)], fill=255)
    draw.ellipse([s(146), s(380), s(208), s(444)], fill=0)
    draw.ellipse([s(592), s(380), s(654), s(444)], fill=0)
    draw.rounded_rectangle([s(214), s(384), s(266), s(432)], radius=s(12), fill=255)
    draw.rounded_rectangle([s(534), s(384), s(586), s(432)], radius=s(12), fill=255)

    # 三足：做成外张的鼎足轮廓
    draw.polygon(
        [(s(288), s(560)), (s(246), s(710)), (s(314), s(710)), (s(338), s(560))],
        fill=255
    )
    draw.polygon(
        [(s(384), s(578)), (s(348), s(732)), (s(452), s(732)), (s(416), s(578))],
        fill=255
    )
    draw.polygon(
        [(s(512), s(560)), (s(486), s(710)), (s(554), s(710)), (s(596), s(560))],
        fill=255
    )

    resample_filter = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
    mask = mask.resize((canvas_size, canvas_size), resample_filter)
    mask = mask.point(lambda pixel: 255 if pixel > 32 else 0)
    # WordCloud 会将白色区域视为不可放置文字，因此这里反相后让文字填充到香炉内部
    return 255 - np.array(mask)

# 创建 mask
mask = create_incense_burner_mask()

# 国潮配色方案
colors = [
    '#C45C48',  # 朱红
    '#3B5C7D',  # 黛蓝
    '#8B4513',  # 赭石
    '#6B8E6B',  # 竹青
    '#D4A574',  # 缃色
    '#8B0000',  # 深红
    '#2F4F4F',  # 深灰蓝
    '#CD853F',  # 秘鲁色
    '#556B2F',  # 深绿
    '#B8860B',  # 暗金
    '#A0522D',  # 赭色
    '#4682B4',  # 钢蓝
]

# 自定义颜色函数
def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return np.random.choice(colors)

# 生成词云
wc = WordCloud(
    font_path=font_path,
    width=800,
    height=800,
    scale=2,
    background_color='white',
    mask=mask,
    max_words=2000,
    relative_scaling=0.2,
    colormap='Reds',  # 基础色调
    prefer_horizontal=1.0,  # 兼容旧版 Pillow，避免竖排文字触发 TransposedFont 报错
    min_font_size=4,
    max_font_size=90,
    font_step=1,
    margin=0,
    repeat=True,
    color_func=color_func,
    collocations=False
).generate(text)

# 创建图形
fig, ax = plt.subplots(figsize=(12, 12), facecolor='white')
ax.imshow(wc, interpolation='bilinear')
ax.axis('off')
ax.set_facecolor('white')

# 添加标题
ax.text(0.5, 0.02, '福州本土国潮香氛消费认知、行为与购买意愿',
        transform=ax.transAxes, fontsize=20, ha='center', va='bottom',
        color='#3B5C7D', fontweight='bold', fontproperties=title_font)

ax.text(0.5, -0.02, 'Fuzhou Local Guochao Fragrance: Perception, Behavior & Purchase Intention',
        transform=ax.transAxes, fontsize=11, ha='center', va='top',
        color='#666666', style='italic')

plt.tight_layout()
output_path = Path(__file__).resolve().parent / 'output' / '图5.8_中式传统香炉词云图.png'
output_path.parent.mkdir(exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', pad_inches=0.2)
if plt.get_backend().lower() != 'agg':
    plt.show()
else:
    plt.close(fig)
print(f"词云图已生成：{output_path}")
