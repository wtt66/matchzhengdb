import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, Wedge
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置样式
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = (12, 8)

# 自定义配色方案 - 国潮风格
COLORS = {
    'primary': ['#C41E3A', '#8B4513', '#D4AF37', '#2F4F4F', '#483D8B'],  # 中国红、赭石、金色、深灰、靛蓝
    'secondary': ['#E8D4C4', '#F5E6D3', '#D4A574', '#8B7355', '#A0522D'],  # 米色、浅棕
    'gradient': ['#FFE4E1', '#FFB6C1', '#FF69B4', '#C41E3A', '#8B0000'],  # 红色渐变
    'channel': ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C', '#E67E22']
}

Q1_COL = '1.您是否了解国潮香氛产品（融合中国传统文化 / 地域元素的香薰、香水、香氛蜡烛等）?【单选】'
Q3_COL = '3.您是否知晓福州的香氛相关非遗技艺（茉莉花窨制技艺、冷凝合香等）?【单选】'
Q4_COL = '4.您是否知晓融入福州地域元素（福文化、榕城、茉莉花等）的国潮香氛产品?【单选】'
CHANNEL_COLS = [
    '2.您了解国潮香氛产品的主要渠道是?【多选】(A. 电商平台（淘宝 / 京东 / 抖音等）)',
    '2 (B. 文旅街区 / 文创店（如三坊七巷、上下杭）)',
    '2 (C. 社交媒体（小红书 / 微博 / 视频号）)',
    '2 (D. 线下商超 / 美妆店)',
    '2 (E. 朋友 / 家人推荐)',
    '2 (F. 酒店 / 民宿等场景体验)',
    '2 (G. 其他)'
]
CHANNEL_NAMES = ['电商平台', '文旅街区', '社交媒体', '线下商超', '亲友推荐', '酒店民宿', '其他']
COGNITION_ORDER = ['完全不了解', '不太了解', '一般了解', '比较了解', '非常了解']
RADAR_CATEGORIES = list(reversed(COGNITION_ORDER))
COGNITION_CODE_TO_LABEL = dict(enumerate(COGNITION_ORDER, start=1))
Q4_CODE_TO_LABEL = {1: '从未知晓', 2: '知晓但未购买', 3: '知晓且购买过'}
HIGH_COGNITION_LEVELS = set(COGNITION_ORDER[2:])
KNOWN_PRODUCT_LEVELS = {'知晓但未购买', '知晓且购买过'}


def normalize_single_choice_answers(series, mapping):
    numeric_values = pd.to_numeric(series, errors='coerce')
    mapped_values = numeric_values.map(mapping)
    return mapped_values.where(~mapped_values.isna(), series)


def normalize_survey_dataframe(df):
    normalized_df = df.copy()
    for col, mapping in (
        (Q1_COL, COGNITION_CODE_TO_LABEL),
        (Q3_COL, COGNITION_CODE_TO_LABEL),
        (Q4_COL, Q4_CODE_TO_LABEL),
    ):
        if col in normalized_df.columns:
            normalized_df[col] = normalize_single_choice_answers(normalized_df[col], mapping)
    return normalized_df


def is_selected(series):
    numeric_selected = pd.to_numeric(series, errors='coerce').eq(1)
    text_values = series.fillna('').astype(str).str.strip().str.lower()
    text_selected = text_values.isin({'是', '选中', 'true', 'yes', 'y'})
    return numeric_selected | text_selected

# ============================================
# 图1: 国潮香氛产品认知度分析 - 雷达图+条形图组合
# ============================================
def plot_cognition_analysis(df, save_path='output/图1_认知度分析.png'):
    """
    深入分析Q1的认知度分布，并与Q3、Q4进行交叉分析
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # 数据准备
    q1_counts = df[Q1_COL].value_counts()
    q3_counts = df[Q3_COL].value_counts()
    q4_counts = df[Q4_COL].value_counts()

    # 标准化认知等级
    cognition_order = COGNITION_ORDER
    q1_ordered = q1_counts.reindex(cognition_order, fill_value=0)
    q3_ordered = q3_counts.reindex(cognition_order, fill_value=0)

    # 子图1: Q1认知度分布 - 3D效果条形图
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.barh(range(len(q1_ordered)), q1_ordered.values,
                    color=COLORS['gradient'], edgecolor='black', linewidth=1.5)
    ax1.set_yticks(range(len(q1_ordered)))
    ax1.set_yticklabels(q1_ordered.index, fontsize=11)
    ax1.set_xlabel('人数', fontsize=12, fontweight='bold')
    ax1.set_title('Q1: 国潮香氛产品认知度分布\n(总体认知水平)', fontsize=13, fontweight='bold', pad=15)

    # 添加数值标签和百分比
    total = q1_ordered.sum()
    for i, (bar, val) in enumerate(zip(bars, q1_ordered.values)):
        width = bar.get_width()
        pct = val / total * 100 if total else 0
        ax1.text(width + 1, bar.get_y() + bar.get_height()/2,
                 f'{val}人 ({pct:.1f}%)',
                 ha='left', va='center', fontsize=10, fontweight='bold')

    # 添加认知度指数计算
    cognition_score = sum([(i+1)*v for i, v in enumerate(q1_ordered.values)]) / total if total else 0
    avg_marker_x = total / 5 if total else 0
    ax1.axvline(x=avg_marker_x, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax1.text(avg_marker_x, -0.5, f'平均认知指数: {cognition_score:.2f}/5',
             fontsize=10, color='red', fontweight='bold', ha='center')

    # 子图2: 认知度雷达图
    ax2 = fig.add_subplot(gs[0, 1], projection='polar')
    categories = RADAR_CATEGORIES
    radar_values = q1_counts.reindex(categories, fill_value=0).values
    values = radar_values / total * 100 if total else np.zeros(len(categories))

    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    values_list = values.tolist()
    values_list += values_list[:1]
    angles += angles[:1]

    ax2.plot(angles, values_list, 'o-', linewidth=3, color=COLORS['primary'][0])
    ax2.fill(angles, values_list, alpha=0.3, color=COLORS['primary'][0])
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories, fontsize=10)
    ax2.set_ylim(0, max(5, float(np.nanmax(values)) * 1.2))
    ax2.set_title('认知度分布雷达图\n(百分比)', fontsize=13, fontweight='bold', pad=20)

    # 子图3: Q1 vs Q3 认知对比
    ax3 = fig.add_subplot(gs[0, 2])
    x = np.arange(len(cognition_order))
    width = 0.35

    bars1 = ax3.bar(x - width/2, q1_ordered.values, width, label='国潮香氛认知',
                    color=COLORS['primary'][0], alpha=0.8, edgecolor='black')
    bars2 = ax3.bar(x + width/2, q3_ordered.values, width, label='福州非遗技艺认知',
                    color=COLORS['primary'][3], alpha=0.8, edgecolor='black')

    ax3.set_xlabel('认知程度', fontsize=12, fontweight='bold')
    ax3.set_ylabel('人数', fontsize=12, fontweight='bold')
    ax3.set_title('Q1 vs Q3: 一般认知 vs 本土认知\n(认知深度对比)', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(cognition_order, rotation=15, ha='right', fontsize=10)
    ax3.legend(fontsize=10, loc='upper left')

    # 添加显著性检验标注
    # 计算卡方检验
    contingency = pd.crosstab(
        df[Q1_COL],
        df[Q3_COL]
    )
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    ax3.text(0.98, 0.95, f'χ² = {chi2:.2f}\np = {p_value:.4f}',
             transform=ax3.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             horizontalalignment='right')

    # 子图4: Q4 知晓度分析 - 饼图+环形图
    ax4 = fig.add_subplot(gs[1, 0])
    q4_labels = ['知晓且购买过', '知晓但未购买', '从未知晓']
    q4_values = [q4_counts.get(label, 0) for label in q4_labels]
    colors_q4 = [COLORS['primary'][0], COLORS['primary'][2], COLORS['primary'][3]]

    wedges, texts, autotexts = ax4.pie(q4_values, labels=q4_labels, autopct='%1.1f%%',
                                       colors=colors_q4, startangle=90,
                                       explode=(0.05, 0.05, 0.1),
                                       shadow=True, textprops={'fontsize': 11})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    ax4.set_title('Q4: 福州本土产品知晓度\n(市场渗透率分析)', fontsize=13, fontweight='bold')

    # 子图5: 认知-知晓转化漏斗
    ax5 = fig.add_subplot(gs[1, 1:])

    # 计算转化数据
    total_sample = len(df)
    know_general = df[Q1_COL].isin(HIGH_COGNITION_LEVELS).sum()
    know_local = df[Q3_COL].isin(HIGH_COGNITION_LEVELS).sum()
    know_product = df[Q4_COL].isin(KNOWN_PRODUCT_LEVELS).sum()
    purchased = (df[Q4_COL] == '知晓且购买过').sum()

    funnel_data = [total_sample, know_general, know_local, know_product, purchased]
    funnel_labels = ['总样本', '了解国潮香氛', '知晓福州非遗', '知晓本土产品', '已购买']
    funnel_colors = ['#2C3E50', '#E74C3C', '#F39C12', '#27AE60', '#8E44AD']

    # 绘制漏斗图
    y_pos = np.arange(len(funnel_data))
    bar_heights = [d/max(funnel_data)*0.8 for d in funnel_data]
    bar_widths = [d/max(funnel_data) for d in funnel_data]

    for i, (h, w, color, label, val) in enumerate(zip(bar_heights, bar_widths, funnel_colors, funnel_labels, funnel_data)):
        left = (1 - w) / 2
        rect = FancyBboxPatch((left, i*0.8), w, 0.6,
                              boxstyle="round,pad=0.02,rounding_size=0.1",
                              facecolor=color, edgecolor='black', linewidth=2, alpha=0.8)
        ax5.add_patch(rect)
        ax5.text(0.5, i*0.8 + 0.3, f'{label}: {val}人 ({val/total_sample*100:.1f}%)',
                 ha='center', va='center', fontsize=11, fontweight='bold', color='white')

        # 添加转化率
        if i > 0:
            conversion = val / funnel_data[i-1] * 100
            ax5.annotate(f'转化率: {conversion:.1f}%',
                         xy=(0.5, i*0.8 + 0.6), xytext=(1.05, i*0.8 + 0.3),
                         fontsize=9, color='darkgreen', fontweight='bold',
                         arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.5))

    ax5.set_xlim(0, 1.5)
    ax5.set_ylim(-0.2, len(funnel_data)*0.8 + 0.5)
    ax5.axis('off')
    ax5.set_title('认知-知晓-购买转化漏斗\n(消费者旅程分析)', fontsize=13, fontweight='bold', pad=20)

    plt.suptitle('第一部分: 国潮香氛产品基础认知深度分析', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()
    print(f"图1已保存至: {save_path}")

    return {
        'cognition_score': cognition_score,
        'conversion_rates': {
            'general_cognition': know_general/total_sample,
            'local_cognition': know_local/total_sample,
            'product_awareness': know_product/total_sample,
            'purchase_rate': purchased/total_sample
        }
    }

# ============================================
# 图2: 信息渠道分析 - 桑基图风格+网络图
# ============================================
def plot_channel_analysis(df, save_path='output/图2_信息渠道分析.png'):
    """
    深入分析Q2的信息渠道，包括渠道组合、主导渠道识别、渠道间关系
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)

    # 渠道列名
    channel_cols = CHANNEL_COLS
    channel_names = CHANNEL_NAMES
    selected_channels = pd.DataFrame({col: is_selected(df[col]).astype(int) for col in channel_cols})

    # 计算各渠道选择人数（处理多选）
    channel_counts = []
    for col in channel_cols:
        count = selected_channels[col].sum()
        channel_counts.append(count)

    channel_df = pd.DataFrame({
        '渠道': channel_names,
        '人数': channel_counts,
        '占比': [c/sum(channel_counts)*100 for c in channel_counts]
    }).sort_values('人数', ascending=True)

    # 子图1: 渠道选择人数 - 水平条形图
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.barh(channel_df['渠道'], channel_df['人数'],
                    color=COLORS['channel'][:len(channel_df)],
                    edgecolor='black', linewidth=1.5, alpha=0.85)
    ax1.set_xlabel('选择人数', fontsize=12, fontweight='bold')
    ax1.set_title('各信息渠道选择人数\n(绝对热度)', fontsize=13, fontweight='bold')

    for bar, val, pct in zip(bars, channel_df['人数'], channel_df['占比']):
        width = bar.get_width()
        ax1.text(width + max(channel_df['人数'])*0.01, bar.get_y() + bar.get_height()/2,
                 f'{val}人 ({pct:.1f}%)', ha='left', va='center', fontsize=10, fontweight='bold')

    # 子图2: 渠道组合分析 - 热力图
    ax2 = fig.add_subplot(gs[0, 1:])

    # 构建渠道共现矩阵
    channel_matrix = np.zeros((len(channel_cols), len(channel_cols)))
    for i, col_i in enumerate(channel_cols):
        for j, col_j in enumerate(channel_cols):
            if i != j:
                # 计算同时选择两个渠道的人数
                mask_i = selected_channels[col_i].eq(1)
                mask_j = selected_channels[col_j].eq(1)
                channel_matrix[i, j] = (mask_i & mask_j).sum()

    # 计算Jaccard相似度
    jaccard_matrix = np.zeros_like(channel_matrix)
    for i in range(len(channel_cols)):
        for j in range(len(channel_cols)):
            if i != j and channel_counts[i] > 0:
                jaccard_matrix[i, j] = channel_matrix[i, j] / (channel_counts[i] + channel_counts[j] - channel_matrix[i, j])

    im = ax2.imshow(jaccard_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax2.set_xticks(range(len(channel_names)))
    ax2.set_yticks(range(len(channel_names)))
    ax2.set_xticklabels(channel_names, rotation=45, ha='right', fontsize=10)
    ax2.set_yticklabels(channel_names, fontsize=10)
    ax2.set_title('渠道组合Jaccard相似度矩阵\n(渠道间关联强度)', fontsize=13, fontweight='bold')

    # 添加数值标注
    for i in range(len(channel_names)):
        for j in range(len(channel_names)):
            if i != j:
                text = ax2.text(j, i, f'{jaccard_matrix[i, j]:.2f}',
                                ha="center", va="center", color="black" if jaccard_matrix[i, j] < 0.5 else "white",
                                fontsize=9, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
    cbar.set_label('Jaccard相似度', fontsize=11, fontweight='bold')

    # 子图3: 人均渠道数分布
    ax3 = fig.add_subplot(gs[1, 0])

    # 计算每个人的渠道数量
    channel_counts_per_person = selected_channels.sum(axis=1).tolist()

    df_temp = pd.DataFrame({'渠道数': channel_counts_per_person})
    channel_dist = df_temp['渠道数'].value_counts().sort_index()

    bars = ax3.bar(channel_dist.index, channel_dist.values,
                   color=COLORS['gradient'][:len(channel_dist)],
                   edgecolor='black', linewidth=1.5, alpha=0.85)
    ax3.set_xlabel('信息渠道数量', fontsize=12, fontweight='bold')
    ax3.set_ylabel('人数', fontsize=12, fontweight='bold')
    ax3.set_title('人均信息渠道数量分布\n(信息获取广度)', fontsize=13, fontweight='bold')

    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}人\n({height/len(df)*100:.1f}%)',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    avg_channels = np.mean(channel_counts_per_person)
    ax3.axvline(x=avg_channels, color='red', linestyle='--', linewidth=2, label=f'平均: {avg_channels:.2f}个')
    ax3.legend(fontsize=10)

    # 子图4: 渠道与认知度的关系
    ax4 = fig.add_subplot(gs[1, 1])

    # 将认知度转化为数值
    cognition_map = {label: idx for idx, label in COGNITION_CODE_TO_LABEL.items()}
    df['认知度数值'] = df[Q1_COL].map(cognition_map)

    channel_cognition = []
    for col, name in zip(channel_cols, channel_names):
        mask = selected_channels[col].eq(1)
        avg_cog = df[mask]['认知度数值'].mean()
        avg_cog = 0 if pd.isna(avg_cog) else avg_cog
        channel_cognition.append(avg_cog)

    channel_cog_df = pd.DataFrame({
        '渠道': channel_names,
        '平均认知度': channel_cognition
    }).sort_values('平均认知度', ascending=True)

    bars = ax4.barh(channel_cog_df['渠道'], channel_cog_df['平均认知度'],
                    color=COLORS['primary'][3], edgecolor='black', linewidth=1.5, alpha=0.8)
    ax4.set_xlim(0, 5)
    ax4.set_xlabel('平均认知度得分', fontsize=12, fontweight='bold')
    ax4.set_title('各渠道用户的平均认知度\n(渠道质量评估)', fontsize=13, fontweight='bold')

    for bar, val in zip(bars, channel_cog_df['平均认知度']):
        width = bar.get_width()
        ax4.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                 f'{val:.2f}', ha='left', va='center', fontsize=10, fontweight='bold')

    # 子图5: 渠道重要性网络图（简化版）
    ax5 = fig.add_subplot(gs[1, 2])

    # 计算各渠道的"中心性"（被共同选择的次数）
    centrality = channel_matrix.sum(axis=1)

    # 绘制气泡图表示渠道重要性
    x_pos = np.random.uniform(0.1, 0.9, len(channel_names))
    y_pos = np.random.uniform(0.1, 0.9, len(channel_names))

    # 调整位置避免重叠（简单力导向）
    for _ in range(50):
        for i in range(len(channel_names)):
            for j in range(i+1, len(channel_names)):
                dx = x_pos[i] - x_pos[j]
                dy = y_pos[i] - y_pos[j]
                dist = np.sqrt(dx**2 + dy**2)
                if dist < 0.2:
                    force = (0.2 - dist) / dist * 0.01
                    x_pos[i] += dx * force
                    y_pos[j] -= dx * force
                    y_pos[i] += dy * force
                    y_pos[j] -= dy * force

    max_centrality = max(float(centrality.max()), 1.0)
    sizes = [c / max_centrality * 2000 + 500 for c in centrality]
    colors_bubble = [plt.cm.RdYlGn(c / max_centrality) for c in centrality]

    scatter = ax5.scatter(x_pos, y_pos, s=sizes, c=colors_bubble, alpha=0.7, edgecolors='black', linewidth=2)

    for i, name in enumerate(channel_names):
        ax5.annotate(name, (x_pos[i], y_pos[i]), fontsize=9, ha='center', va='center', fontweight='bold')

    # 绘制连线（共现关系）
    non_zero_links = channel_matrix[channel_matrix > 0]
    threshold = np.percentile(non_zero_links, 75) if len(non_zero_links) else np.inf
    for i in range(len(channel_names)):
        for j in range(i+1, len(channel_names)):
            if channel_matrix[i, j] > threshold:
                ax5.plot([x_pos[i], x_pos[j]], [y_pos[i], y_pos[j]],
                         'k-', alpha=0.3, linewidth=channel_matrix[i, j] / max(float(channel_matrix.max()), 1.0) * 5)

    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    ax5.axis('off')
    ax5.set_title('渠道关联网络图\n(气泡大小=中心性,连线=强共现)', fontsize=13, fontweight='bold')

    plt.suptitle('第二部分: 信息渠道深度分析', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()
    print(f"图2已保存至: {save_path}")

    return {
        'channel_ranking': channel_df.to_dict('records'),
        'avg_channels_per_person': avg_channels,
        'channel_cognition_correlation': channel_cog_df.to_dict('records')
    }

# ============================================
# 图3: 非遗与地域认知交叉分析 - 马赛克图+聚类
# ============================================
def plot_heritage_local_analysis(df, save_path='output/图3_非遗地域认知分析.png'):
    """
    深入分析Q3和Q4的关系，识别不同认知群体特征
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    q3_col = Q3_COL
    q4_col = Q4_COL

    # 子图1: Q3 vs Q4 交叉分析热力图
    ax1 = fig.add_subplot(gs[0, 0])

    cross_tab = pd.crosstab(df[q3_col], df[q4_col], normalize='index') * 100

    im = ax1.imshow(cross_tab.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    ax1.set_xticks(range(len(cross_tab.columns)))
    ax1.set_yticks(range(len(cross_tab.index)))
    ax1.set_xticklabels(cross_tab.columns, rotation=15, ha='right', fontsize=10)
    ax1.set_yticklabels(cross_tab.index, fontsize=10)
    ax1.set_xlabel('Q4: 本土产品知晓度', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Q3: 非遗技艺认知度', fontsize=12, fontweight='bold')
    ax1.set_title('非遗认知 vs 产品知晓 交叉分析\n(条件概率 %)', fontsize=13, fontweight='bold')

    for i in range(len(cross_tab.index)):
        for j in range(len(cross_tab.columns)):
            text = ax1.text(j, i, f'{cross_tab.iloc[i, j]:.1f}%',
                            ha="center", va="center",
                            color="white" if cross_tab.iloc[i, j] > 50 else "black",
                            fontsize=11, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
    cbar.set_label('百分比 (%)', fontsize=11, fontweight='bold')

    # 子图2: 认知群体细分
    ax2 = fig.add_subplot(gs[0, 1])

    # 定义认知群体
    def classify_group(row):
        q3 = row[q3_col]
        q4 = row[q4_col]

        high_q3 = q3 in ['非常了解', '比较了解']
        high_q4 = q4 in ['知晓且购买过', '知晓但未购买']

        if high_q3 and high_q4:
            return '深度认知者'
        elif high_q3 and not high_q4:
            return '非遗关注者'
        elif not high_q3 and high_q4:
            return '产品关注者'
        else:
            return '潜在启蒙者'

    df['认知群体'] = df.apply(classify_group, axis=1)
    group_counts = df['认知群体'].value_counts()

    colors_group = ['#C41E3A', '#D4AF37', '#2F4F4F', '#8B7355']
    explode = (0.1, 0.05, 0.05, 0.05)

    wedges, texts, autotexts = ax2.pie(group_counts.values, labels=group_counts.index,
                                       autopct='%1.1f%%', colors=colors_group,
                                       explode=explode, shadow=True, startangle=90,
                                       textprops={'fontsize': 11})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)

    ax2.set_title('认知群体细分\n(基于非遗与产品认知交叉)', fontsize=13, fontweight='bold')

    # 添加图例说明
    legend_text = (
        '深度认知者: 了解非遗且知晓产品\n'
        '非遗关注者: 了解非遗但未关注产品\n'
        '产品关注者: 知晓产品但不了解非遗\n'
        '潜在启蒙者: 两者均不了解'
    )
    ax2.text(1.3, 0.5, legend_text, transform=ax2.transAxes, fontsize=9,
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 子图3: 各群体的信息渠道偏好对比
    ax3 = fig.add_subplot(gs[1, 0])

    channel_cols = CHANNEL_COLS
    channel_names = CHANNEL_NAMES

    group_channel_pref = []
    for group in group_counts.index:
        group_df = df[df['认知群体'] == group]
        prefs = []
        for col in channel_cols:
            pct = is_selected(group_df[col]).mean() * 100
            prefs.append(pct)
        group_channel_pref.append(prefs)

    x = np.arange(len(channel_names))
    width = 0.2

    for i, (group, prefs) in enumerate(zip(group_counts.index, group_channel_pref)):
        offset = (i - 1.5) * width
        bars = ax3.bar(x + offset, prefs, width, label=group,
                       color=colors_group[i], alpha=0.8, edgecolor='black')

    ax3.set_xlabel('信息渠道', fontsize=12, fontweight='bold')
    ax3.set_ylabel('选择比例 (%)', fontsize=12, fontweight='bold')
    ax3.set_title('不同认知群体的信息渠道偏好\n(群体差异化分析)', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(channel_names, rotation=30, ha='right', fontsize=10)
    ax3.legend(fontsize=9, loc='upper right')

    # 子图4: 认知深度与购买行为的关系
    ax4 = fig.add_subplot(gs[1, 1])

    # 分析各群体的购买转化率
    purchase_conversion = []
    for group in group_counts.index:
        group_df = df[df['认知群体'] == group]
        purchased = (group_df[q4_col] == '知晓且购买过').sum()
        total = len(group_df)
        conversion = purchased / total * 100 if total > 0 else 0
        purchase_conversion.append(conversion)

    bars = ax4.bar(group_counts.index, purchase_conversion,
                   color=colors_group, edgecolor='black', linewidth=2, alpha=0.85)
    ax4.set_ylabel('购买转化率 (%)', fontsize=12, fontweight='bold')
    ax4.set_title('各认知群体的购买转化率\n(认知-行为转化效率)', fontsize=13, fontweight='bold')
    ax4.set_ylim(0, max(purchase_conversion) * 1.2)

    for bar, val, count in zip(bars, purchase_conversion, group_counts.values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{val:.1f}%\n(n={count})',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 添加趋势线
    z = np.polyfit(range(len(purchase_conversion)), purchase_conversion, 1)
    p = np.poly1d(z)
    ax4.plot(range(len(purchase_conversion)), p(range(len(purchase_conversion))),
             "r--", alpha=0.8, linewidth=2, label='趋势')
    ax4.legend(fontsize=10)

    plt.suptitle('第三部分: 非遗与地域认知深度分析', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()
    print(f"图3已保存至: {save_path}")

    return {
        'cognitive_groups': group_counts.to_dict(),
        'purchase_conversion_by_group': dict(zip(group_counts.index, purchase_conversion)),
        'cross_tab': cross_tab.to_dict()
    }

# ============================================
# 图4: 综合认知画像 - 主成分分析+聚类可视化
# ============================================
def plot_comprehensive_profile(df, save_path='output/图4_综合认知画像.png'):
    """
    构建综合认知画像，进行受访者聚类分析
    """
    from sklearn.preprocessing import LabelEncoder
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)

    # 数据编码
    cognition_map = {label: idx for idx, label in COGNITION_CODE_TO_LABEL.items()}
    q4_map = {label: idx for idx, label in Q4_CODE_TO_LABEL.items()}

    df['Q1编码'] = df[Q1_COL].map(cognition_map)
    df['Q3编码'] = df[Q3_COL].map(cognition_map)
    df['Q4编码'] = df[Q4_COL].map(q4_map)

    # 渠道数量
    selected_channels = pd.DataFrame({col: is_selected(df[col]).astype(int) for col in CHANNEL_COLS})
    df['渠道数量'] = selected_channels.sum(axis=1)

    # 特征矩阵
    features = df[['Q1编码', 'Q3编码', 'Q4编码', '渠道数量']].fillna(0)

    # 子图1: PCA降维可视化
    ax1 = fig.add_subplot(gs[0, :2])

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features)

    # K-means聚类
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(features)
    df['认知聚类'] = clusters

    colors_cluster = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
    for i in range(4):
        mask = clusters == i
        ax1.scatter(pca_result[mask, 0], pca_result[mask, 1],
                    c=colors_cluster[i], label=f'聚类 {i+1}',
                    alpha=0.6, s=100, edgecolors='black', linewidth=0.5)

    # 绘制聚类中心
    centers = pca.transform(kmeans.cluster_centers_)
    ax1.scatter(centers[:, 0], centers[:, 1], c='black', marker='X', s=300,
                label='聚类中心', edgecolors='white', linewidth=2, zorder=5)

    ax1.set_xlabel(f'第一主成分 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12, fontweight='bold')
    ax1.set_ylabel(f'第二主成分 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12, fontweight='bold')
    ax1.set_title('受访者认知画像PCA聚类分析\n(基于Q1/Q3/Q4/渠道数量的无监督聚类)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)

    # 添加特征向量
    feature_names = ['国潮认知', '非遗认知', '产品知晓', '渠道数量']
    for i, (name, vec) in enumerate(zip(feature_names, pca.components_.T)):
        ax1.annotate('', xy=(vec[0]*3, vec[1]*3), xytext=(0, 0),
                     arrowprops=dict(arrowstyle='->', color='red', lw=2, alpha=0.7))
        ax1.text(vec[0]*3.2, vec[1]*3.2, name, fontsize=10, color='red', fontweight='bold')

    # 子图2: 聚类特征雷达图
    ax2 = fig.add_subplot(gs[0, 2], projection='polar')

    cluster_profiles = df.groupby('认知聚类')[['Q1编码', 'Q3编码', 'Q4编码', '渠道数量']].mean()
    # 标准化到0-5分制
    cluster_profiles_norm = cluster_profiles.copy()
    cluster_profiles_norm['渠道数量'] = cluster_profiles_norm['渠道数量'] / cluster_profiles_norm['渠道数量'].max() * 5

    categories = ['国潮认知', '非遗认知', '产品知晓', '渠道广度']
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    for i, (cluster, values) in enumerate(cluster_profiles_norm.iterrows()):
        values_list = values.tolist()
        values_list += values_list[:1]
        ax2.plot(angles, values_list, 'o-', linewidth=2.5, label=f'聚类{cluster+1}', color=colors_cluster[i])
        ax2.fill(angles, values_list, alpha=0.15, color=colors_cluster[i])

    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories, fontsize=11)
    ax2.set_ylim(0, 5)
    ax2.set_title('各聚类认知特征雷达图\n(标准化得分)', fontsize=13, fontweight='bold', pad=20)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

    # 子图3: 聚类统计详情
    ax3 = fig.add_subplot(gs[1, 0])

    cluster_stats = df['认知聚类'].value_counts().sort_index()
    cluster_pct = cluster_stats / len(df) * 100

    bars = ax3.bar([f'聚类{i+1}' for i in cluster_stats.index], cluster_stats.values,
                   color=colors_cluster, edgecolor='black', linewidth=2, alpha=0.85)
    ax3.set_ylabel('人数', fontsize=12, fontweight='bold')
    ax3.set_title('各聚类样本分布', fontsize=13, fontweight='bold')

    for bar, count, pct in zip(bars, cluster_stats.values, cluster_pct.values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{count}人\n({pct:.1f}%)',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 子图4: 聚类与购买行为的关系
    ax4 = fig.add_subplot(gs[1, 1])

    purchase_by_cluster = pd.crosstab(df['认知聚类'],
                                      df[Q4_COL],
                                      normalize='index') * 100

    purchase_by_cluster.plot(kind='bar', stacked=True, ax=ax4,
                             color=['#E74C3C', '#F39C12', '#27AE60'],
                             edgecolor='black', linewidth=1.5)
    ax4.set_xlabel('认知聚类', fontsize=12, fontweight='bold')
    ax4.set_ylabel('比例 (%)', fontsize=12, fontweight='bold')
    ax4.set_title('各聚类的购买行为分布\n(知晓-购买转化)', fontsize=13, fontweight='bold')
    ax4.set_xticklabels([f'聚类{i+1}' for i in range(4)], rotation=0)
    ax4.legend(title='产品知晓度', fontsize=9, title_fontsize=10)

    # 子图5: 综合认知指数分布
    ax5 = fig.add_subplot(gs[1, 2])

    # 计算综合认知指数 (加权平均)
    df['综合认知指数'] = (df['Q1编码'] * 0.3 + df['Q3编码'] * 0.3 + df['Q4编码'] * 0.4)

    # 绘制箱线图和小提琴图组合
    parts = ax5.violinplot([df[df['认知聚类']==i]['综合认知指数'].values for i in range(4)],
                           positions=range(4), showmeans=False, showmedians=False, showextrema=False)

    for pc, color in zip(parts['bodies'], colors_cluster):
        pc.set_facecolor(color)
        pc.set_alpha(0.3)
        pc.set_edgecolor('black')
        pc.set_linewidth(1.5)

    bp = ax5.boxplot([df[df['认知聚类']==i]['综合认知指数'].values for i in range(4)],
                     positions=range(4), widths=0.3, patch_artist=True,
                     boxprops=dict(facecolor='white', alpha=0.8, edgecolor='black', linewidth=1.5),
                     medianprops=dict(color='red', linewidth=2),
                     whiskerprops=dict(color='black', linewidth=1.5),
                     capprops=dict(color='black', linewidth=1.5))

    ax5.set_xticks(range(4))
    ax5.set_xticklabels([f'聚类{i+1}' for i in range(4)])
    ax5.set_ylabel('综合认知指数', fontsize=12, fontweight='bold')
    ax5.set_title('各聚类综合认知指数分布\n(箱线图+小提琴图)', fontsize=13, fontweight='bold')
    ax5.set_ylim(0, 5.5)

    # 添加均值线
    for i in range(4):
        mean_val = df[df['认知聚类']==i]['综合认知指数'].mean()
        ax5.scatter(i, mean_val, color='red', s=100, zorder=5, marker='D')
        ax5.text(i, mean_val + 0.15, f'{mean_val:.2f}', ha='center', fontsize=10, fontweight='bold', color='red')

    plt.suptitle('第四部分: 综合认知画像与聚类分析', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()
    print(f"图4已保存至: {save_path}")

    # 输出聚类标签建议
    cluster_labels = {
        0: "启蒙型消费者",
        1: "文化深度型",
        2: "产品导向型",
        3: "全面认知型"
    }

    return {
        'cluster_profiles': cluster_profiles.to_dict(),
        'cluster_sizes': cluster_stats.to_dict(),
        'pca_explained_variance': pca.explained_variance_ratio_.tolist(),
        'suggested_labels': cluster_labels
    }

# ============================================
# 主执行函数
# ============================================
def main():
    """
    主执行函数：读取数据并生成所有分析图表
    """
    # 创建输出目录
    import os
    os.makedirs('output', exist_ok=True)

    # 读取数据 - 请根据实际路径修改
    file_path = 'E:\\python\\zhengdabei\\data\\第一部分.csv'

    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except:
        try:
            df = pd.read_csv(file_path, encoding='gbk')
        except:
            df = pd.read_csv(file_path, encoding='gb2312')

    df = normalize_survey_dataframe(df)

    print(f"成功读取数据，样本量: {len(df)}")
    print(f"变量数: {len(df.columns)}")
    print("\n开始生成分析图表...")

    # 生成四张核心分析图
    results = {}

    print("\n[1/4] 生成认知度分析图...")
    results['cognition'] = plot_cognition_analysis(df)

    print("\n[2/4] 生成信息渠道分析图...")
    results['channel'] = plot_channel_analysis(df)

    print("\n[3/4] 生成非遗地域认知分析图...")
    results['heritage'] = plot_heritage_local_analysis(df)

    print("\n[4/4] 生成综合认知画像图...")
    results['profile'] = plot_comprehensive_profile(df)

    print("\n" + "="*60)
    print("所有图表生成完成！")
    print("="*60)
    print("\n关键发现摘要:")
    print(f"1. 总体认知指数: {results['cognition']['cognition_score']:.2f}/5.0")
    print(f"2. 平均信息渠道数: {results['channel']['avg_channels_per_person']:.2f}个")
    print(f"3. 购买转化率: {results['cognition']['conversion_rates']['purchase_rate']*100:.1f}%")
    print(f"4. 主要认知群体: {max(results['heritage']['cognitive_groups'], key=results['heritage']['cognitive_groups'].get)}")

    return results

# 如果直接运行此脚本
if __name__ == "__main__":
    results = main()
