import streamlit as st
import PyPDF2
import re
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# 常见同义词表
SYNONYMS = {
    'nlp': ['自然语言处理', 'NLP'],
    'python': ['Python', 'python'],
    'java': ['Java', 'java'],
    'sql': ['SQL', 'sql', '结构化查询语言'],
    '机器学习': ['ML', '机器学习', 'machine learning'],
    '深度学习': ['DL', '深度学习', 'deep learning'],
    '数据分析': ['数据分析师', '数据分析', 'data analysis'],
    '产品经理': ['PM', '产品经理', 'product manager'],
    '项目管理': ['项目管理', 'project management'],
    '团队合作': ['团队协作', '团队合作', 'teamwork'],
    '沟通能力': ['沟通', '沟通能力', 'communication'],
    '领导力': ['领导', '领导力', 'leadership'],
    '抗压能力': ['抗压', '抗压能力', 'stress resistance']
}

# 教育层次映射
EDUCATION_LEVELS = {
    '高中': 1,
    '大专': 2,
    '本科': 3,
    '硕士': 4,
    '博士': 5
}

# 从PDF提取文本
def extract_text_from_pdf(pdf_file):
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ''
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text() or ''
        return text
    except Exception as e:
        st.error(f"PDF解析失败: {str(e)}")
        return None

# 提取关键词
def extract_keywords(text, top_n=20):
    # 分词
    words = jieba.cut(text)
    words = [word for word in words if len(word) > 1]
    text_processed = ' '.join(words)
    
    # TF-IDF提取关键词
    vectorizer = TfidfVectorizer(max_features=top_n)
    try:
        vectorizer.fit([text_processed])
        keywords = vectorizer.get_feature_names_out()
        return list(keywords)
    except:
        return []

# 分类关键词
def categorize_keywords(keywords):
    hard_skills = []
    experience_terms = []
    education_terms = []
    soft_skills = []
    
    # 硬技能关键词
    hard_skill_patterns = r'\b(python|java|sql|c\+\+|javascript|html|css|react|vue|angular|node|php|go|rust|mysql|postgresql|mongodb|redis|docker|kubernetes|aws|azure|gcp|linux|git|github|jira|figma|photoshop|illustrator|excel|powerpoint|word|tableau|powerbi|tensorflow|pytorch|keras|scikit-learn|pandas|numpy|matplotlib|seaborn|nlp|自然语言处理|机器学习|深度学习|数据分析|数据挖掘|算法|架构|后端|前端|全栈|移动开发|ios|android|测试|自动化|devops|ci/cd)\b'
    
    # 经验相关关键词
    experience_patterns = r'\b(经验|年|工作|项目|职责|负责|参与|主导|管理|开发|设计|实现|优化|部署|维护|运营|市场|销售|客户|用户|产品|需求|规划|执行|监控|分析|报告)\b'
    
    # 教育相关关键词
    education_patterns = r'\b(本科|硕士|博士|大专|高中|学历|专业|毕业|学位|院校|大学|学院|专业|计算机|软件|电子|通信|数学|统计|经济|管理|营销|中文|英语)\b'
    
    # 软技能关键词
    soft_skill_patterns = r'\b(团队|合作|沟通|领导力|抗压|创新|学习|适应|解决问题|时间管理|组织|协调|表达|写作|演讲|逻辑|分析|思维|执行|责任|积极|主动|热情|专业|诚信|可靠)\b'
    
    for keyword in keywords:
        if re.search(hard_skill_patterns, keyword, re.IGNORECASE):
            hard_skills.append(keyword)
        elif re.search(experience_patterns, keyword):
            experience_terms.append(keyword)
        elif re.search(education_patterns, keyword):
            education_terms.append(keyword)
        elif re.search(soft_skill_patterns, keyword):
            soft_skills.append(keyword)
    
    return {
        'hard_skills': hard_skills,
        'experience_terms': experience_terms,
        'education_terms': education_terms,
        'soft_skills': soft_skills
    }

# 计算硬技能匹配度
def calculate_hard_skill_match(resume_text, jd_keywords):
    if not jd_keywords['hard_skills']:
        return 100.0
    
    match_count = 0
    resume_lower = resume_text.lower()
    
    for skill in jd_keywords['hard_skills']:
        # 精确匹配
        if skill.lower() in resume_lower:
            match_count += 1
        else:
            # 同义词匹配
            for syn_list in SYNONYMS.values():
                if skill.lower() in [s.lower() for s in syn_list]:
                    for syn in syn_list:
                        if syn.lower() in resume_lower:
                            match_count += 1
                            break
                    break
    
    return (match_count / len(jd_keywords['hard_skills'])) * 100

# 计算经验匹配度
def calculate_experience_match(resume_text, jd_text):
    # 提取工作年限
    years_match = re.search(r'(\d+)\s*年', resume_text)
    years_of_experience = int(years_match.group(1)) if years_match else 0
    
    # 从JD中提取经验要求
    jd_years_match = re.search(r'(\d+)\s*年', jd_text)
    required_years = int(jd_years_match.group(1)) if jd_years_match else 0
    
    # 计算年限匹配度
    if required_years == 0:
        years_score = 100.0
    else:
        years_score = min(100.0, (years_of_experience / required_years) * 100)
    
    # 提取职位名称相似度（简化处理）
    resume_positions = re.findall(r'([\u4e00-\u9fa5a-zA-Z]+)\s*[、,，]?', resume_text)
    jd_positions = re.findall(r'([\u4e00-\u9fa5a-zA-Z]+)\s*[、,，]?', jd_text)
    
    position_score = 0.0
    if resume_positions and jd_positions:
        # 简单的字符串匹配
        for resume_pos in resume_positions:
            for jd_pos in jd_positions:
                if resume_pos in jd_pos or jd_pos in resume_pos:
                    position_score = 100.0
                    break
            if position_score == 100.0:
                break
    
    # 项目描述相似度（关键词匹配）
    project_keywords = ['项目', '负责', '主导', '参与', '开发', '设计', '实现', '优化']
    project_score = 0.0
    if any(keyword in resume_text for keyword in project_keywords):
        project_score = 100.0
    
    # 综合经验得分
    return (years_score * 0.4 + position_score * 0.3 + project_score * 0.3)

# 计算教育背景匹配度
def calculate_education_match(resume_text, jd_text):
    # 提取简历中的最高学历
    resume_education = 0
    for edu, level in EDUCATION_LEVELS.items():
        if edu in resume_text:
            if level > resume_education:
                resume_education = level
    
    # 提取JD中的最低学历要求
    jd_education = 0
    for edu, level in EDUCATION_LEVELS.items():
        if edu in jd_text:
            if level > jd_education:
                jd_education = level
    
    # 学历匹配度
    if jd_education == 0:
        education_score = 100.0
    elif resume_education >= jd_education:
        education_score = 100.0
    else:
        education_score = 0.0
    
    # 专业匹配度（简化处理）
    major_keywords = ['计算机', '软件', '电子', '通信', '数学', '统计', '经济', '管理', '营销']
    major_score = 0.0
    if any(keyword in resume_text for keyword in major_keywords):
        major_score = 100.0
    
    # 综合教育得分
    return (education_score * 0.7 + major_score * 0.3)

# 计算软技能匹配度
def calculate_soft_skill_match(resume_text, jd_keywords):
    if not jd_keywords['soft_skills']:
        return 100.0
    
    match_count = 0
    resume_lower = resume_text.lower()
    
    for skill in jd_keywords['soft_skills']:
        if skill.lower() in resume_lower:
            match_count += 1
        else:
            # 同义词匹配
            for syn_list in SYNONYMS.values():
                if skill.lower() in [s.lower() for s in syn_list]:
                    for syn in syn_list:
                        if syn.lower() in resume_lower:
                            match_count += 1
                            break
                    break
    
    return (match_count / len(jd_keywords['soft_skills'])) * 100

# 主函数
def main():
    st.title("HR 简历-岗位匹配度分析工具")
    
    # 上传JD
    st.subheader("1. 岗位描述 (JD)")
    jd_input_option = st.radio("选择输入方式:", ("直接输入", "上传文件"))
    
    jd_text = ""
    if jd_input_option == "直接输入":
        jd_text = st.text_area("请粘贴岗位描述:", height=200)
    else:
        jd_file = st.file_uploader("上传JD文件 (支持 .txt, .docx)", type=["txt", "docx"])
        if jd_file:
            if jd_file.name.endswith('.txt'):
                jd_text = jd_file.read().decode('utf-8')
            else:
                st.warning("暂时仅支持 .txt 文件，其他格式请使用直接输入方式")
    
    # 上传简历
    st.subheader("2. 求职者简历")
    resume_file = st.file_uploader("上传简历 PDF 文件:", type=["pdf"])
    
    # 权重设置
    st.subheader("3. 匹配维度权重")
    hard_skill_weight = st.slider("硬技能匹配权重", 0, 100, 40, 5)
    experience_weight = st.slider("经验匹配权重", 0, 100, 30, 5)
    education_weight = st.slider("教育背景权重", 0, 100, 15, 5)
    soft_skill_weight = st.slider("软技能与附加项权重", 0, 100, 15, 5)
    
    # 确保权重总和为100
    total_weight = hard_skill_weight + experience_weight + education_weight + soft_skill_weight
    if total_weight != 100:
        st.warning(f"权重总和应为100%，当前为{total_weight}%，将自动调整")
        # 简单调整：按比例分配
        hard_skill_weight = int(hard_skill_weight / total_weight * 100)
        experience_weight = int(experience_weight / total_weight * 100)
        education_weight = int(education_weight / total_weight * 100)
        soft_skill_weight = 100 - hard_skill_weight - experience_weight - education_weight
        st.info(f"调整后权重: 硬技能={hard_skill_weight}%, 经验={experience_weight}%, 教育={education_weight}%, 软技能={soft_skill_weight}%")
    
    # 分析按钮
    if st.button("开始分析匹配度"):
        if not jd_text:
            st.error("请输入岗位描述")
            return
        
        if not resume_file:
            st.error("请上传简历PDF文件")
            return
        
        # 提取简历文本
        st.info("正在解析简历...")
        resume_text = extract_text_from_pdf(resume_file)
        
        if not resume_text:
            st.error("无法从PDF中提取文本，请确保上传的是文字版PDF")
            return
        
        # 提取JD关键词并分类
        st.info("正在分析岗位要求...")
        jd_keywords = extract_keywords(jd_text)
        categorized_keywords = categorize_keywords(jd_keywords)
        
        # 计算各维度得分
        st.info("正在计算匹配度...")
        hard_skill_score = calculate_hard_skill_match(resume_text, categorized_keywords)
        experience_score = calculate_experience_match(resume_text, jd_text)
        education_score = calculate_education_match(resume_text, jd_text)
        soft_skill_score = calculate_soft_skill_match(resume_text, categorized_keywords)
        
        # 计算总匹配度
        total_match = (
            hard_skill_score * hard_skill_weight / 100 +
            experience_score * experience_weight / 100 +
            education_score * education_weight / 100 +
            soft_skill_score * soft_skill_weight / 100
        )
        
        # 显示结果
        st.subheader("4. 匹配度分析结果")
        
        # 环形进度条
        st.markdown(f"""
        <div style="text-align: center; margin: 20px 0;">
            <div style="display: inline-block; position: relative; width: 200px; height: 200px;">
                <svg width="200" height="200" viewBox="0 0 100 100">
                    <!-- 背景圆 -->
                    <circle cx="50" cy="50" r="45" fill="none" stroke="#e6e6e6" stroke-width="8"/>
                    <!-- 进度圆 -->
                    <circle cx="50" cy="50" r="45" fill="none" stroke="{'#4CAF50' if total_match >= 80 else '#FF9800'}" stroke-width="8" 
                            stroke-dasharray="{2 * 3.14159 * 45}" 
                            stroke-dashoffset="{2 * 3.14159 * 45 * (1 - total_match / 100)}" 
                            transform="rotate(-90 50 50)" stroke-linecap="round"/>
                </svg>
                <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-size: 32px; font-weight: bold;">
                    {total_match:.1f}%
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 匹配度结论
        if total_match >= 80:
            st.success("✅ 建议邀约线下面试")
        else:
            st.warning("⚠️ 匹配度不足")
            
            # 找出得分最低的维度
            scores = {
                '硬技能匹配': hard_skill_score,
                '经验匹配': experience_score,
                '教育背景': education_score,
                '软技能与附加项': soft_skill_score
            }
            lowest_dimension = min(scores, key=scores.get)
            lowest_score = scores[lowest_dimension]
            
            st.info(f"主要短板: {lowest_dimension} (得分: {lowest_score:.1f}%)")
            
            # 给出改进建议
            if lowest_dimension == '硬技能匹配':
                st.info("建议: 加强相关技术技能的学习和实践，在简历中突出相关技能经验")
            elif lowest_dimension == '经验匹配':
                st.info("建议: 增加相关工作经验，突出与岗位相关的项目经历和职责")
            elif lowest_dimension == '教育背景':
                st.info("建议: 考虑提升学历层次，或强调与岗位相关的专业技能和知识")
            else:
                st.info("建议: 加强软技能的培养，在简历中突出团队合作、沟通等能力")
        
        # 各维度得分明细
        st.subheader("5. 各维度得分明细")
        st.table({
            '维度': ['硬技能匹配', '经验匹配', '教育背景', '软技能与附加项'],
            '得分': [f"{hard_skill_score:.1f}%", f"{experience_score:.1f}%", f"{education_score:.1f}%", f"{soft_skill_score:.1f}%"],
            '权重': [f"{hard_skill_weight}%", f"{experience_weight}%", f"{education_weight}%", f"{soft_skill_weight}%"]
        })

if __name__ == "__main__":
    main()