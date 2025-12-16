import streamlit as st
import pandas as pd
import altair as alt
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 页面基础配置
# ==========================================
st.set_page_config(
    page_title="电商评论观点挖掘系统",
    page_icon="📊",
    layout="wide"
)

# 设置 Matplotlib 中文字体 (防止词云乱码)
# 尝试常见的中文字体，如果是在 Linux 服务器上可能需要指定特定路径
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

# 18个评价维度定义
LABEL_COLUMNS = [
    'Location#Transportation', 'Location#Downtown', 'Location#Easy_to_find',
    'Service#Queue', 'Service#Hospitality', 'Service#Parking', 'Service#Timely',
    'Price#Level', 'Price#Cost_effective', 'Price#Discount',
    'Ambience#Decoration', 'Ambience#Noise', 'Ambience#Space', 'Ambience#Sanitary',
    'Food#Portion', 'Food#Taste', 'Food#Appearance', 'Food#Recommend'
]

# 维度中文映射 (让图表更好看)
ASPECT_MAP = {
    'Food#Taste': '味道/口感', 'Food#Portion': '分量', 'Food#Appearance': '外观', 'Food#Recommend': '总体推荐',
    'Price#Level': '价格水平', 'Price#Cost_effective': '性价比', 'Price#Discount': '折扣优惠',
    'Service#Timely': '物流/时效', 'Service#Hospitality': '服务态度', 'Service#Queue': '排队',
    'Ambience#Decoration': '包装/环境', 'Ambience#Sanitary': '卫生',
    # ... 其他不太重要的可以保留原名或继续补充
}


# ==========================================
# 2. 数据加载函数
# ==========================================
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        # 确保时间列是 datetime 格式
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
        return df
    except Exception as e:
        return None


# ==========================================
# 3. 侧边栏 (Sidebar)
# ==========================================
st.sidebar.title("🛠️ 系统控制面板")

# 文件上传/选择
uploaded_file = st.sidebar.file_uploader("上传分析结果 CSV", type=['csv'])
default_file = "final_analysis_result.csv"  # 默认读取的文件名

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.sidebar.success("✅ 已加载上传文件")
elif os.path.exists(default_file):
    df = load_data(default_file)
    st.sidebar.info(f"ℹ️ 已加载默认文件: {default_file}")
else:
    st.error("未找到数据文件！请上传 CSV 或确保目录下有 final_analysis_result.csv")
    st.stop()

# 侧边栏筛选器
st.sidebar.subheader("数据筛选")
if 'time' in df.columns:
    min_date = df['time'].min().date() if pd.notnull(df['time'].min()) else None
    max_date = df['time'].max().date() if pd.notnull(df['time'].max()) else None
    if min_date and max_date:
        start_date, end_date = st.sidebar.date_input("选择时间范围", [min_date, max_date])

# ==========================================
# 4. 主界面 (Main Dashboard)
# ==========================================
st.title("电商商品评论观点挖掘与分析系统")
st.markdown("基于 BERT 深度学习模型的细粒度情感分析结果展示")

# --- 第一部分：关键指标 (KPI) ---
st.subheader("1. 关键数据概览")
col1, col2, col3, col4 = st.columns(4)

total_comments = len(df)
# 计算总体好评率 (基于 Food#Recommend 或 Score)
if 'score' in df.columns:
    positive_rate = (df[df['score'] >= 4].shape[0] / total_comments) * 100
    metric_label = "五星好评率"
else:
    # 如果没有 score，简单统计 Food#Recommend 为正面的比例
    pos_rec = df[df['Food#Recommend'] == '正面'].shape[0]
    positive_rate = (pos_rec / total_comments) * 100
    metric_label = "推荐指数 (基于模型)"

with col1:
    st.metric("总评论数", f"{total_comments} 条")
with col2:
    st.metric(metric_label, f"{positive_rate:.1f}%")
with col3:
    # 统计提及最多的维度
    counts = {}
    for col in LABEL_COLUMNS:
        counts[col] = df[df[col] != '未提及'].shape[0]
    top_aspect = max(counts, key=counts.get)
    st.metric("最热讨论点", ASPECT_MAP.get(top_aspect, top_aspect))
with col4:
    st.metric("模型分析维度", "18 个")

st.divider()

# --- 第二部分：多维情感分析图表 ---
st.subheader("2. 属性维度情感分布")

# 数据预处理：转换为适合绘图的长格式 (Long Format)
plot_data = []
for col in LABEL_COLUMNS:
    # 统计每个维度的 正面/负面/中性 数量
    vc = df[col].value_counts()
    for sentiment in ['正面', '负面', '中性']:
        count = vc.get(sentiment, 0)
        if count > 0:
            plot_data.append({
                '维度': ASPECT_MAP.get(col, col),  # 使用中文名
                '原始维度': col,
                '情感': sentiment,
                '评论数': count
            })

df_plot = pd.DataFrame(plot_data)

# 交互式堆叠柱状图 (Stacked Bar Chart)
chart = alt.Chart(df_plot).mark_bar().encode(
    x=alt.X('维度', sort='-y'),
    y='评论数',
    color=alt.Color('情感', scale=alt.Scale(domain=['正面', '中性', '负面'], range=['#28a745', '#ffc107', '#dc3545'])),
    tooltip=['维度', '情感', '评论数']
).properties(height=400).interactive()

st.altair_chart(chart, use_container_width=True)

st.info("💡 **图表解读**：绿色代表正面评价，红色代表负面评价。柱子越高，代表用户讨论该属性的次数越多。")

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("3. 用户关注点排行")
    # 统计各维度提及总次数
    aspect_counts = df_plot.groupby('维度')['评论数'].sum().reset_index().sort_values('评论数', ascending=False)

    bar_chart = alt.Chart(aspect_counts).mark_bar().encode(
        x=alt.X('评论数'),
        y=alt.Y('维度', sort='-x'),
        color=alt.value('#3182bd')
    ).properties(height=300)
    st.altair_chart(bar_chart, use_container_width=True)

with col_b:
    st.subheader("4. 负面评价重灾区")
    # 只看负面
    neg_counts = df_plot[df_plot['情感'] == '负面'].sort_values('评论数', ascending=False).head(5)

    if not neg_counts.empty:
        neg_chart = alt.Chart(neg_counts).mark_bar().encode(
            x=alt.X('维度', sort='-y'),
            y='评论数',
            color=alt.value('#dc3545')  # 红色
        ).properties(height=300)
        st.altair_chart(neg_chart, use_container_width=True)
    else:
        st.success("暂无显著的负面评价聚集！")

st.divider()

# --- 第三部分：词云分析 ---
st.subheader("5. 消费者观点词云")
st.write("展示评论中出现频率最高的词汇（基于评论正文）。")

# 选择查看正面还是负面词云
wc_option = st.radio("选择词云类型", ["全部", "仅看好评", "仅看差评"], horizontal=True)

text_content = ""
if wc_option == "仅看好评":
    # 简单筛选：Score >=4 或 含有正面标签
    text_content = " ".join(df[df['score'] >= 4]['content'].astype(str)) if 'score' in df.columns else " ".join(
        df['content'].astype(str))
elif wc_option == "仅看差评":
    text_content = " ".join(df[df['score'] <= 2]['content'].astype(str)) if 'score' in df.columns else " ".join(
        df['content'].astype(str))
else:
    text_content = " ".join(df['content'].astype(str))

if text_content:

    try:
        font_path = 'Arial Unicode.ttf'

        wc = WordCloud(
            width=800, height=400,
            background_color='white',
            font_path=font_path  # 直接用同级目录下的文件名
        ).generate(text_content)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

    except Exception as e:
        st.error(f" 词云生成出错: {e}")
st.divider()

# --- 第四部分：数据透视与下载 ---
st.subheader("6. 原始数据查询")
st.write("您可以查看每一条评论的模型分析结果。")

# 交互式表格
st.dataframe(df, use_container_width=True)

# 下载按钮
csv = df.to_csv(index=False).encode('utf-8-sig')
st.download_button(
    "📥 下载分析报告 (CSV)",
    csv,
    "analysis_report.csv",
    "text/csv",
    key='download-csv'
)