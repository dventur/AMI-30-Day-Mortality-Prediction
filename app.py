import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image
import base64
from io import BytesIO
import shap  # 添加shap库导入

# 修复NumPy bool弃用问题
import numpy as np
if not hasattr(np, 'bool'):
    np.bool = bool

# 设置页面标题和布局
st.set_page_config(
    page_title="AMI ICU 30-Day Mortality Prediction System",
    page_icon="🏥",
    layout="wide"
)

# 只加载模型，不加载解释器
@st.cache_resource
def load_model():
    model = joblib.load('RF.pkl')
    return model

# 加载特征名称和英文对照
feature_names = [
    'Age', 'ICU losday', 'HR', 'SBP', 'RR', 'Temperature', 'SPO2', 
    'Paraplegia', 'Metastatic solid tumor', 'SAPSII', 'WBC', 'Chloride', 
    'BUN', 'Glucose', 'Creatinine', 'Urineoutput', 'PT', 'Bicarbonate', 
    'Aniongap', 'RDW'
]

feature_names_en = [
    'Age', 'ICU Length of Stay', 'Heart Rate', 'Systolic Blood Pressure', 'Respiratory Rate', 'Temperature', 'Oxygen Saturation',
    'Paraplegia', 'Metastatic Solid Tumor', 'SAPS II Score', 'White Blood Cell Count', 'Chloride',
    'Blood Urea Nitrogen', 'Glucose', 'Creatinine', 'Urine Output', 'Prothrombin Time', 'Bicarbonate',
    'Anion Gap', 'Red Cell Distribution Width'
]

feature_dict = dict(zip(feature_names, feature_names_en))

# 变量说明字典
variable_descriptions = {
    'Age': 'Patient age in years',
    'ICU losday': 'ICU length of stay, indicating the duration of patient stay in intensive care unit',
    'HR': 'Heart rate, number of heartbeats per minute, normal range 60-100 bpm',
    'SBP': 'Systolic blood pressure, pressure when heart contracts, normal range 90-140 mmHg',
    'RR': 'Respiratory rate, number of breaths per minute, normal range 12-20 breaths/min',
    'Temperature': 'Body temperature, normal range 36.5-37.5°C',
    'SPO2': 'Oxygen saturation, normal value ≥95%',
    'Paraplegia': 'Paraplegia, indicates whether the patient has paraplegia (0=No, 1=Yes)',
    'Metastatic solid tumor': 'Metastatic solid tumor, indicates whether the patient has metastatic tumor (0=No, 1=Yes)',
    'SAPSII': 'Simplified Acute Physiology Score II, scoring system to assess severity of ICU patients',
    'WBC': 'White blood cell count, assesses infection and inflammation, normal range 4-10×10^9/L',
    'Chloride': 'Chloride ion, electrolyte indicator, normal range 98-106 mmol/L',
    'BUN': 'Blood urea nitrogen, assesses kidney function, normal range 7-20 mg/dL',
    'Glucose': 'Blood glucose, normal range 70-110 mg/dL',
    'Creatinine': 'Creatinine, assesses kidney function, normal range 0.5-1.2 mg/dL',
    'Urineoutput': 'Urine output, daily urine volume, normal adult about 1500-2000 mL/day',
    'PT': 'Prothrombin time, assesses coagulation function, normal range 11-15 seconds',
    'Bicarbonate': 'Bicarbonate, assesses acid-base balance, normal range 22-26 mmol/L',
    'Aniongap': 'Anion gap, assesses acid-base balance, normal range 8-16 mmol/L',
    'RDW': 'Red cell distribution width, assesses red blood cell size variation, normal range 11.5-14.5%'
}

# 主应用
def main():
    # 侧边栏标题
    st.sidebar.title("AMI ICU 30-Day Mortality Prediction System")
    st.sidebar.image("https://img.freepik.com/free-vector/hospital-logo-design-vector-medical-cross_53876-136743.jpg", width=200)
    
    # 添加系统说明到侧边栏
    st.sidebar.markdown("""
    # System Description

    ## About This System
    This is an AMI (Acute Myocardial Infarction) ICU 30-day mortality prediction system based on Random Forest algorithm, which predicts mortality risk by analyzing patient clinical indicators.

    ## Prediction Results
    The system predicts the patient's 30-day:
    - Survival probability
    - Mortality probability
    - Risk assessment (low, medium, high risk)

    ## How to Use
    1. Fill in patient clinical indicators in the main interface
    2. Click the prediction button to generate prediction results
    3. View prediction results and feature importance analysis

    ## Important Notes
    - Please ensure accurate patient information input
    - All fields need to be filled
    - Numeric fields require number input
    - Selection fields require choosing from options
    """)
    
    # 添加变量说明到侧边栏
    with st.sidebar.expander("Variable Descriptions"):
        for feature in feature_names:
            st.markdown(f"**{feature_dict[feature]}**: {variable_descriptions[feature]}")
    
    # 主页面标题
    st.title("AMI ICU 30-Day Mortality Prediction System")
    st.markdown("### Based on Random Forest Model")
    
    # 加载模型
    try:
        model = load_model()
        st.sidebar.success("Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Model loading failed: {e}")
        return
    
    # 创建输入表单
    st.sidebar.header("Patient Information Input")
    
    # 创建两列布局用于输入
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Basic Information")
        age = st.number_input(f"{feature_dict['Age']} (years)", min_value=18, max_value=100, value=65)
        icu_losday = st.number_input(f"{feature_dict['ICU losday']} (days)", min_value=0.1, max_value=30.0, value=3.0, step=0.1)
        hr = st.number_input(f"{feature_dict['HR']} (bpm)", min_value=40, max_value=200, value=80)
        sbp = st.number_input(f"{feature_dict['SBP']} (mmHg)", min_value=60, max_value=200, value=120)
        rr = st.number_input(f"{feature_dict['RR']} (breaths/min)", min_value=8, max_value=40, value=18)
        temperature = st.number_input(f"{feature_dict['Temperature']} (°C)", min_value=35.0, max_value=40.0, value=36.8, step=0.1)
        spo2 = st.number_input(f"{feature_dict['SPO2']} (%)", min_value=70, max_value=100, value=96)
        paraplegia = st.selectbox(f"{feature_dict['Paraplegia']}", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        metastatic = st.selectbox(f"{feature_dict['Metastatic solid tumor']}", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        sapsii = st.number_input(f"{feature_dict['SAPSII']}", min_value=0, max_value=100, value=30)
    
    with col2:
        st.subheader("Laboratory Tests")
        wbc = st.number_input(f"{feature_dict['WBC']} (×10^9/L)", min_value=0.1, max_value=50.0, value=10.0, step=0.1)
        chloride = st.number_input(f"{feature_dict['Chloride']} (mmol/L)", min_value=70, max_value=130, value=100)
        bun = st.number_input(f"{feature_dict['BUN']} (mg/dL)", min_value=5, max_value=150, value=20)
        glucose = st.number_input(f"{feature_dict['Glucose']} (mg/dL)", min_value=50, max_value=500, value=120)
        creatinine = st.number_input(f"{feature_dict['Creatinine']} (mg/dL)", min_value=0.3, max_value=10.0, value=1.0, step=0.1)
        urineoutput = st.number_input(f"{feature_dict['Urineoutput']} (mL/day)", min_value=0, max_value=2000, value=200)
        pt = st.number_input(f"{feature_dict['PT']} (seconds)", min_value=10.0, max_value=60.0, value=14.0, step=0.1)
        bicarbonate = st.number_input(f"{feature_dict['Bicarbonate']} (mmol/L)", min_value=10, max_value=40, value=22)
        aniongap = st.number_input(f"{feature_dict['Aniongap']} (mmol/L)", min_value=5, max_value=30, value=15)
        rdw = st.number_input(f"{feature_dict['RDW']} (%)", min_value=10.0, max_value=25.0, value=14.0, step=0.1)
    
    # 创建预测按钮
    predict_button = st.button("Predict Mortality Risk")
    
    if predict_button:
        # 收集所有输入特征
        features = [age, icu_losday, hr, sbp, rr, temperature, spo2, paraplegia, metastatic, 
                   sapsii, wbc, chloride, bun, glucose, creatinine, urineoutput, pt, 
                   bicarbonate, aniongap, rdw]
        
        # 转换为DataFrame
        input_df = pd.DataFrame([features], columns=feature_names)
        
        # 进行预测
        prediction = model.predict_proba(input_df)[0]
        survival_prob = prediction[0]
        death_prob = prediction[1]
        
        # 显示预测结果
        st.header("Prediction Results")
        
        # 使用进度条显示概率
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Survival Probability")
            st.progress(float(survival_prob))
            st.write(f"{survival_prob:.2%}")
        
        with col2:
            st.subheader("Mortality Probability")
            st.progress(float(death_prob))
            st.write(f"{death_prob:.2%}")
        
        # 风险评估
        risk_level = "Low Risk" if death_prob < 0.3 else "Medium Risk" if death_prob < 0.6 else "High Risk"
        risk_color = "green" if death_prob < 0.3 else "orange" if death_prob < 0.6 else "red"
        
        st.markdown(f"### Risk Assessment: <span style='color:{risk_color}'>{risk_level}</span>", unsafe_allow_html=True)
        
        # 临床建议
        st.header("Clinical Recommendations")
        st.write("Based on the model prediction, the following clinical recommendations are provided:")
        
        if death_prob > 0.5:
            st.warning("This patient has a high mortality risk. Close monitoring and intensive treatment are recommended.")
        else:
            st.success("This patient has a relatively good prognosis. Standard treatment protocol is recommended.")
        
        # 添加SHAP值解释
        st.write("---")
        st.subheader("Model Interpretation")
        
        try:
            # 计算SHAP值
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)

            if isinstance(shap_values, list) and len(shap_values) > 1:
                base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
                shap_value = shap_values[1][0]
            else:
                base_value = explainer.expected_value
                shap_value = shap_values[0]

            # 修正 base_value 和 shap_value 结构
            if isinstance(base_value, (list, np.ndarray)):
                base_value = base_value[0]
            if isinstance(shap_value, np.ndarray) and shap_value.ndim == 2 and shap_value.shape[0] == 1:
                shap_value = shap_value.flatten()

            # input_df.iloc[0] 必须是 Series
            plt.figure(figsize=(15, 4))
            shap.force_plot(
                base_value,
                shap_value,
                input_df.iloc[0],
                matplotlib=True,
                show=False
            )
            plt.tight_layout()
            st.pyplot(plt)
            plt.close()

            # 显示特征重要性说明
            st.write("---")
            st.subheader("Feature Contribution Analysis")
            feature_importance = pd.DataFrame({
                'Feature': input_df.columns,
                'SHAP Value': np.abs(shap_value)
            }).sort_values('SHAP Value', ascending=False)
            st.table(feature_importance)

        except Exception as e:
            st.error(f"无法生成SHAP解释: {str(e)}")
            st.info("使用模型的特征重要性作为替代")
            feature_importance = pd.DataFrame({
                'Feature': input_df.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            st.table(feature_importance)

if __name__ == "__main__":
    main()
