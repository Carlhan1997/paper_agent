import requests
import feedparser
import smtplib
from email.mime.text import MIMEText
import re
import json
import os
import datetime
from openai import OpenAI


# ==========================================
# 1. 抓取 arXiv 凝聚态最新文章
# ==========================================
def fetch_arxiv_cm():
    # 获取最新的150篇凝聚态文章（基本能覆盖当天的更新量）
    url = "http://export.arxiv.org/api/query?search_query=cat:cond-mat.*&start=0&max_results=150&sortBy=submittedDate&sortOrder=descending"
    
    print("正在连接 arXiv 获取数据，请稍候（国内网络可能需要几秒钟）...")
    try:
        response = requests.get(url, timeout=15)
        feed = feedparser.parse(response.text)
    except Exception as e:
        print(f"网络连接失败，请检查网络或代理设置。错误: {e}")
        return []

    papers = []
    for entry in feed.entries:
        # 清洗数据：去掉arXiv摘要里自带的换行符和HTML标签
        raw_summary = entry.summary
        clean_summary = re.sub(r'<[^>]+>', '', raw_summary).replace('\n', ' ').strip()
        
        paper = {
            "title": entry.title.strip(),
            "summary": clean_summary,
            "url": entry.link
        }
        papers.append(paper)
    
    print(f"成功抓取到 {len(papers)} 篇文章，开始进行关键词筛选...")
    return papers

# ==========================================
# 1.5 抓取顶尖期刊的最新文章 (通过CrossRef)
# ==========================================
def fetch_top_journals():
    # 顶级期刊的名称（必须与CrossRef数据库中的名称完全一致）及对应的简称
    JOURNALS = {
        "Nature": "Nat",
        "Nature Physics": "Nat Phys",
        "Nature Materials": "Nat Mater",
        "Nature Communications": "Nat Commun",
        "Physical Review Letters": "PRL",
        "Science": "Science"
    }
    
    # 获取昨天的日期，格式必须为 YYYY-MM-DD
    yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    
    all_journal_papers = []
    
    print("正在连接 CrossRef 获取顶尖期刊数据...")
    for journal_name, abbr in JOURNALS.items():
        # 构造 CrossRef API URL
        url = f"https://api.crossref.org/journals/{journal_name}/works?filter=from-pub-date:{yesterday},until-pub-date:{yesterday}&rows=20"
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                continue
                
            data = response.json()
            items = data["message"]["items"]
            
            for item in items:
                # 提取标题
                title = item.get("title", [""])[0] if item.get("title") else "无标题"
                # 提取摘要 (CrossRef的摘要通常带有HTML/JATS标签，需要用正则清理)
                raw_abstract = item.get("abstract", "无摘要")
                clean_abstract = re.sub(r'<[^>]+>', '', raw_abstract).strip()
                
                # 提取链接（优先使用官方DOI链接）
                doi = item.get("DOI", "")
                url_link = f"https://doi.org/{doi}" if doi else "无链接"
                
                all_journal_papers.append({
                    "title": f"[{abbr}] {title}", # 在标题前加上期刊简称，方便你一眼看出出处
                    "summary": clean_abstract,
                    "url": url_link
                })
            print(f"  -> 成功获取 {journal_name} 的数据")
            
        except Exception as e:
            print(f"  -> 获取 {journal_name} 失败: {e}")
            
    print(f"顶尖期刊共抓取到 {len(all_journal_papers)} 篇文章。")
    return all_journal_papers


# ==========================================
# 2. 关键词筛选与分类 (方案A)
# ==========================================
# ==========================================
# 2. 大模型智能筛选与分类 (方案B)
# ==========================================
def filter_and_classify(papers):
    # ------------------------------------------
    # ★★★ 大模型配置 ★★★
    # ------------------------------------------
    # 如果你用其他模型（如阿里通义、硅基流动），只需修改 base_url 和 api_key 即可
    client = OpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY"), 
        base_url="https://api.deepseek.com"
    )
    MODEL_NAME = "deepseek-chat" # 使用的模型名称
    
    # 定义你期望的分类类别
    CATEGORIES = [
        "拓扑量子材料", 
        "超导物理", 
        "二维材料与莫尔", 
        "磁学与自旋电子"
        "扫描隧道显微镜"
    ]
    
    # 构造给大模型的提示词（Prompt非常关键，决定了分类准确度）
    prompt_template = """你是一个凝聚态物理领域的资深专家。请阅读下面的论文标题和摘要（来源包含 arXiv 预印本及 Nature、Science、PRL 等顶级期刊）。请将其归入以下类别之一：
1. 拓扑量子材料
2. 超导物理
3. 二维材料与莫尔
4. 磁学与自旋电子
5. 扫描隧道显微镜

【判断标准】：
- 只看论文的核心物理内容，不要因为摘要里顺带提了一句其他概念就误判。
- 如果这篇论文不属于上述4个物理方向（比如是做生物物理、软物质、复杂系统的），请归类为：其他。

【严格规则】：
只返回类别名称本身，不要返回任何标点符号、解释文字或前缀。例如：直接返回“超导物理”。

论文标题：{title}
论文摘要：{summary}"""

    categorized = {k: [] for k in CATEGORIES}
    
    print("正在调用大模型进行智能分类，这大概需要一分钟，请耐心等待...")
    
    for i, paper in enumerate(papers):
        # 填充提示词
        prompt = prompt_template.format(
            title=paper["title"], 
            summary=paper["summary"][:800] # 截断过长的摘要，省钱省时间
        )
        
        try:
            # 调用大模型接口
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, # 温度设低，保证分类结果稳定
                max_tokens=10     # 只需要返回几个字，tokens设小点
            )
            
            # 获取大模型返回的结果，并清理掉可能带上的标点符号
            result = response.choices[0].message.content.strip().replace("。", "").replace(".", "")
            
            # 打印进度，让你知道程序没卡死
            print(f"[{i+1}/{len(papers)}] {paper['title'][:30]}... -> 归类为: {result}")
            
            # 如果大模型返回的分类在我们的字典里，就存进去
            if result in categorized:
                categorized[result].append(paper)
                
        except Exception as e:
            print(f"[{i+1}/{len(papers)}] 调用API失败，跳过该文章。错误: {e}")
            continue

    # 过滤掉空分类
    final_result = {k: v for k, v in categorized.items() if len(v) > 0}
    return final_result


# ==========================================
# 3. 构建并发送邮件
# ==========================================
def send_email(categorized_papers):
    
    # ------------------------------------------
    # ★★★ 邮件配置（在这里填入你自己的信息） ★★★
    # ------------------------------------------
    SMTP_SERVER = "smtp.163.com"     # 如果是QQ邮箱，改成 "smtp.qq.com"
    SMTP_PORT = 465
    SENDER_EMAIL = os.environ.get("SENDER_EMAIL")
    SENDER_AUTH_CODE = os.environ.get("SENDER_AUTH_CODE")
    RECEIVER_EMAIL = os.environ.get("RECEIVER_EMAIL")
    # ------------------------------------------

    today = datetime.datetime.now().strftime('%Y-%m-%d')
    
    # 拼接 HTML 邮件正文
    html_content = f"""
    <div style="font-family: 'Microsoft YaHei', Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px;">
        <h2 style="color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px;">
            📅 arXiv 凝聚态每日速递 ({today})
        </h2>
    """
    
    has_paper = False
    for category, papers in categorized_papers.items():
        has_paper = True
        html_content += f"""
        <div style="margin-top: 25px;">
            <h3 style="color: #e74c3c;">🔴 {category} ({len(papers)}篇)</h3>
            <ul style="list-style-type: none; padding-left: 5px;">
        """
        for p in papers:
            # 只截取前150个字符作为摘要预览，避免邮件太长
            short_summary = p["summary"][:150] + "..."
            html_content += f"""
                <li style="margin-bottom: 15px; padding-left: 10px; border-left: 3px solid #ecf0f1;">
                    <a href="{p['url']}" style="color: #2980b9; text-decoration: none; font-weight: bold; font-size: 15px;">
                        {p['title']}
                    </a>
                    <br>
                    <span style="color: #7f8c8d; font-size: 13px; line-height: 1.5;">
                        {short_summary}
                    </span>
                </li>
            """
        html_content += "</ul></div>"
    
    if not has_paper:
        html_content += "<p style='color: #95a5a6;'>今天暂无符合你关键词的文章，去休息一下吧！</p>"
    
    html_content += "</div>"

    # 设置邮件头和正文
    msg = MIMEText(html_content, "html", "utf-8")
    msg["Subject"] = f"arXiv 凝聚态日报 {today}"
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL

    # 发送邮件
    print("正在登录邮箱服务器...")
    try:
        server = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT)
        server.login(SENDER_EMAIL, SENDER_AUTH_CODE)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        server.quit()
        print("✅ 邮件发送成功！快去收件箱查看吧。")
    except Exception as e:
        print(f"❌ 邮件发送失败！请检查邮箱配置或授权码是否正确。错误信息: {e}")

# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    # 1. 抓取 arXiv 数据
    arxiv_papers = fetch_arxiv_cm()
    
    # 2. 抓取 顶尖期刊 数据
    journal_papers = fetch_top_journals()
    
    # 3. 合并两个数据源！(Python中合并列表非常简单，用 + 号即可)
    all_papers = arxiv_papers + journal_papers
    
    if len(all_papers) == 0:
        print("未抓取到任何数据，程序结束。")
    else:
        print(f"\n★★★ 总计抓取 {len(all_papers)} 篇文章，准备送入大模型进行智能分类 ★★★\n")
        # 4. 大模型筛选 (不管是哪来的论文，大模型一视同仁地进行语义分类)
        result = filter_and_classify(all_papers)
        # 5. 发送邮件
        send_email(result)
    
    # 运行完毕后暂停
    input("程序执行完毕，按回车键退出...")

