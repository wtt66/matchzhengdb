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

# 福州国潮香氛消费痛点数据（词语:权重）
pain_points_freq = {
    # 核心痛点（最大）
    "留香短": 100,
    "伴手礼同质化": 95,
    "性价比低": 88,
    "文化融合浅": 85,

    # 产品体验痛点
    "香型单一": 72,
    "扩香差": 68,
    "持香短": 65,
    "层次单薄": 60,
    "前调刺鼻": 58,
    "中调缺失": 55,
    "尾调消散快": 52,

    # 价格与价值痛点
    "溢价严重": 70,
    "价格虚高": 65,
    "不值定价": 62,
    "促销依赖": 55,
    "套装捆绑": 50,
    "替换装贵": 48,

    # 文化认同痛点
    "文化符号堆砌": 68,
    "故事空洞": 62,
    "地域特色弱": 58,
    "传统现代割裂": 55,
    "IP联名敷衍": 52,
    "非遗浮于表面": 48,

    # 购买决策痛点
    "试香不便": 60,
    "线上线下差异": 55,
    "包装过度": 52,
    "运输破损": 48,
    "售后缺失": 45,
    "真假难辨": 42,

    # 使用场景痛点
    "场景局限": 58,
    "季节性强": 52,
    "空间适配差": 48,
    "人群覆盖窄": 45,
    "礼品属性重": 42,
    "自用尴尬": 38,

    # 品牌认知痛点
    "品牌认知低": 55,
    "本土信任弱": 50,
    "国际对比自卑": 45,
    "营销过度": 42,
    "KOL滤镜重": 38,
    "UGC真实性存疑": 35,

    # 渠道与服务痛点
    "线下网点少": 48,
    "体验空间缺": 45,
    "专业导购少": 42,
    "定制服务无": 38,
    "会员体系弱": 35,
    "社群运营差": 32
}

# 创建古典香囊形状的 mask（800x800）
def create_sachet_mask():
    canvas_size = 800
    scale = 3
    mask = Image.new('L', (canvas_size * scale, canvas_size * scale), 0)
    draw = ImageDraw.Draw(mask)

    def s(value):
        return int(value * scale)

    # 顶部挂环：保留传统悬挂感，但不做成炉钮样式
    draw.ellipse([s(344), s(28), s(456), s(132)], fill=255)
    draw.ellipse([s(374), s(58), s(426), s(104)], fill=0)
    draw.rounded_rectangle([s(388), s(106), s(412), s(160)], radius=s(8), fill=255)

    # 盘扣与绳结
    draw.polygon([(s(320), s(140)), (s(370), s(112)), (s(392), s(154)), (s(350), s(188))], fill=255)
    draw.polygon([(s(480), s(140)), (s(430), s(112)), (s(408), s(154)), (s(450), s(188))], fill=255)
    draw.ellipse([s(364), s(138), s(436), s(206)], fill=255)
    draw.rounded_rectangle([s(310), s(192), s(490), s(238)], radius=s(20), fill=255)

    # 香囊主体：束口收紧、袋身鼓起、底部下坠
    body_outline = [
        (s(318), s(228)),
        (s(266), s(252)),
        (s(222), s(314)),
        (s(194), s(406)),
        (s(202), s(500)),
        (s(238), s(582)),
        (s(302), s(654)),
        (s(366), s(708)),
        (s(400), s(732)),
        (s(434), s(708)),
        (s(498), s(654)),
        (s(562), s(582)),
        (s(598), s(500)),
        (s(606), s(406)),
        (s(578), s(314)),
        (s(534), s(252)),
        (s(482), s(228)),
    ]
    draw.polygon(body_outline, fill=255)
    draw.ellipse([s(188), s(236), s(612), s(552)], fill=255)
    draw.ellipse([s(226), s(308), s(574), s(678)], fill=255)
    draw.rounded_rectangle([s(286), s(214), s(514), s(272)], radius=s(22), fill=255)

    # 底部流苏：只保留中间流苏，避免像香炉双耳
    draw.rounded_rectangle([s(364), s(720), s(436), s(756)], radius=s(10), fill=255)
    draw.polygon([(s(348), s(748)), (s(318), s(798)), (s(360), s(798)), (s(386), s(748))], fill=255)
    draw.polygon([(s(390), s(748)), (s(380), s(800)), (s(420), s(800)), (s(410), s(748))], fill=255)
    draw.polygon([(s(414), s(748)), (s(440), s(748)), (s(482), s(798)), (s(452), s(798))], fill=255)

    resample_filter = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
    mask = mask.resize((canvas_size, canvas_size), resample_filter)
    mask = mask.point(lambda pixel: 255 if pixel > 32 else 0)
    # WordCloud 会将白色区域视为不可放置文字，因此这里反相后让文字填充到香囊内部
    return 255 - np.array(mask)

# 创建 mask
mask = create_sachet_mask()

# 古典香囊配色 - 深红、绛紫、黛蓝、赭石、暗金（痛点用偏暗色调）
sachet_colors = [
    '#8B0000',  # 暗红（最严重痛点）
    '#660033',  # 绛紫
    '#4A0E4E',  # 深紫
    '#8B4513',  # 赭石
    '#654321',  # 深褐
    '#2F4F4F',  # 深灰蓝
    '#800020',  # 勃艮第红
    '#5C4033',  # 深棕
    '#8B7355',  # 暖褐
    '#A0522D',  # 赭色
    '#483D8B',  # 暗岩蓝
    '#6B4423',  # 咖啡棕
]

# 自定义颜色函数 - 大词用深色，小词用相对浅色
def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    if font_size > 60:
        return np.random.choice(['#8B0000', '#660033', '#4A0E4E', '#800020'])
    elif font_size > 40:
        return np.random.choice(['#8B4513', '#654321', '#2F4F4F', '#5C4033'])
    else:
        return np.random.choice(['#8B7355', '#A0522D', '#483D8B', '#6B4423'])

# 生成词云
wc = WordCloud(
    font_path=font_path,
    width=800,
    height=800,
    scale=2,
    background_color='white',
    mask=mask,
    max_words=1800,
    relative_scaling=0.22,
    colormap='Reds',
    prefer_horizontal=1.0,  # 兼容旧版 Pillow，避免竖排文字触发 TransposedFont 报错
    min_font_size=4,
    max_font_size=100,
    font_step=1,
    margin=0,
    repeat=True,
    color_func=color_func,
    collocations=False
).generate_from_frequencies(pain_points_freq)

# 创建图形
fig, ax = plt.subplots(figsize=(12, 12), facecolor='white')
fig.subplots_adjust(top=0.90, bottom=0.10)
ax.imshow(wc, interpolation='bilinear')
ax.axis('off')
ax.set_facecolor('white')

# 添加标题
ax.text(0.5, 0.02, '福州本土国潮香氛消费痛点分析',
        transform=ax.transAxes, fontsize=22, ha='center', va='bottom',
        color='#660033', fontweight='bold', fontproperties=title_font)

ax.text(0.5, -0.02, 'Fuzhou Local Guochao Fragrance: Consumer Pain Points',
        transform=ax.transAxes, fontsize=11, ha='center', va='top',
        color='#666666', style='italic')

# 添加图例说明核心痛点
legend_text = "核心痛点：留香短 | 伴手礼同质化 | 性价比低 | 文化融合浅"
fig.text(0.5, 0.965, legend_text,
         fontsize=12, ha='center', va='top',
         color='#8B0000', fontweight='bold', fontproperties=title_font,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF8DC',
                   edgecolor='#8B4513', alpha=0.8))

plt.tight_layout(rect=(0.03, 0.08, 0.97, 0.90))
output_path = Path(__file__).resolve().parent / 'output' / '古典香囊痛点词云图.png'
output_path.parent.mkdir(exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', pad_inches=0.2)
if plt.get_backend().lower() != 'agg':
    plt.show()
else:
    plt.close(fig)
print(f"痛点词云图已生成：{output_path}")
