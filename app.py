import streamlit as st
import joblib
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from skimage.feature import graycomatrix, graycoprops

# 0. 전역 변수 설정 (에러 방지를 위해 함수 밖으로 꺼냄)
FEATURE_ORDER = [
    'H_mean', 'S_mean', 'V_mean', 'G_mean', 'B_mean', 
    'Color_Variance', 'Contrast_0', 'Contrast_45', 'Contrast_135', 
    'Homogeneity_45', 'Homogeneity_90', 'Homogeneity_135', 
    'Correlation_avg', 'Energy_avg', 'ASM_avg', 'Entropy_avg', 
    'Gray_Mean_Center', 'Texture_Complexity', 'Uniformity_Score', 'Edge_Density'
]

# 1. 모델 및 데이터 로드
@st.cache_resource
def load_assets():
    model = joblib.load('finals_tuned_xgboost_20feats.pkl') 
    scaler = joblib.load('scaler_20feats.pkl')
    # 혼동 행렬 데이터가 별도로 없다면 학습 시 결과를 수동으로 넣거나 파일을 로드합니다.
    # 여기서는 예시로 학습 시 성능 데이터를 세팅합니다.
    return model, scaler

model, scaler = load_assets()

# 2. 특징 추출 함수
def extract_20_features(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    features = {}
    
    # 색상 특징
    features['H_mean'], features['S_mean'], features['V_mean'] = np.mean(hsv[:,:,0]), np.mean(hsv[:,:,1]), np.mean(hsv[:,:,2])
    features['G_mean'], features['B_mean'] = np.mean(img_array[:,:,1]), np.mean(img_array[:,:,2])
    features['Color_Variance'] = np.var(img_array)
    
    # 질감 특징 (GLCM)
    glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    features['Contrast_0'] = graycoprops(glcm, 'contrast')[0, 0]
    features['Contrast_45'] = graycoprops(glcm, 'contrast')[0, 1]
    features['Contrast_135'] = graycoprops(glcm, 'contrast')[0, 3]
    features['Homogeneity_45'] = graycoprops(glcm, 'homogeneity')[0, 1]
    features['Homogeneity_90'] = graycoprops(glcm, 'homogeneity')[0, 2]
    features['Homogeneity_135'] = graycoprops(glcm, 'homogeneity')[0, 3]
    features['Correlation_avg'] = np.mean(graycoprops(glcm, 'correlation'))
    features['Energy_avg'] = np.mean(graycoprops(glcm, 'energy'))
    features['ASM_avg'] = np.mean(graycoprops(glcm, 'ASM'))
    features['Entropy_avg'] = -np.sum(glcm * np.log2(glcm + 1e-10))
    
    # 추가 특징
    h, w = gray.shape
    features['Gray_Mean_Center'] = np.mean(gray[h//2-16:h//2+16, w//2-16:w//2+16])
    features['Texture_Complexity'] = np.std(graycoprops(glcm, 'contrast'))
    features['Uniformity_Score'] = np.mean(graycoprops(glcm, 'homogeneity'))
    features['Edge_Density'] = np.mean(cv2.Canny(gray, 100, 200)) / 255.0

    ordered_values = [features.get(name, 0.0) for name in FEATURE_ORDER]
    return np.array(ordered_values).reshape(1, -1)

# --- UI 구성 ---
st.set_page_config(page_title="암 진단 AI 리포트", layout="wide")
st.title("🔬 AI Pathology Report")

uploaded_file = st.file_uploader("조직 이미지(TIF) 업로드", type=['tif', 'tiff'])

if uploaded_file:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.image(image, caption="분석 이미지", use_container_width=True)

    with col2:
        extracted_data = extract_20_features(img_array)
        scaled_data = scaler.transform(extracted_data)
        prediction = model.predict(scaled_data)[0]
        prob = model.predict_proba(scaled_data)[0][1]

        st.subheader("📋 판독 결과")
        if prediction == 1:
            st.error(f"### 🚨 암 조직 가능성 높음 ({prob*100:.2f}%)")
            st.progress(float(prob)) # float로 형변환하여 에러 해결
        else:
            st.success(f"### ✅ 정상 조직 가능성 높음 ({(1-prob)*100:.2f}%)")
            st.progress(float(1-prob))

    st.divider()
    
    # --- 여기서부터 판단 근거 (시각화) ---
    st.header("📊 AI 판단 근거 분석 (Model Explainability)")
    
    tab1, tab2, tab3 = st.tabs(["✨ 특징 중요도", "🧪 혼동 행렬", "📝 상세 수치"])
    
    with tab1:
        st.write("모델이 판독 시 가장 중요하게 생각하는 TOP 10 요인입니다.")
        importances = model.feature_importances_
        fi_df = pd.DataFrame({'Feature': FEATURE_ORDER, 'Importance': importances})
        fi_df = fi_df.sort_values('Importance', ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=fi_df, x='Importance', y='Feature', palette='magma', ax=ax)
        plt.title("Top 10 Feature Importance (Gain)")
        st.pyplot(fig)

    with tab2:
        st.write("학습 시 검증 세트를 통해 측정된 모델의 신뢰도 지표입니다.")
        
        # 1. 컬럼을 생성하여 중간 컬럼의 너비를 조절합니다.
        # [1, 1, 1]은 1:1:1 비율이며, 숫자를 조절해 크기를 더 줄일 수 있습니다.
        col1, col2, col3 = st.columns([1, 1.2, 1]) 

        with col2: # 가운데 칸에만 그래프를 넣습니다.
            cm_data = [[19463, 1537], [1269, 19731]] 
            
            # 2. figsize도 함께 줄여주면 텍스트 비율이 더 잘 맞습니다.
            fig_cm, ax_cm = plt.subplots(figsize=(4, 3.2)) 
            
            sns.heatmap(cm_data, annot=True, fmt='d', cmap='RdPu', 
                        xticklabels=['Normal', 'Cancer'], yticklabels=['Normal', 'Cancer'],
                        annot_kws={"size": 10}) # 글자 크기도 살짝 조절
            
            plt.xlabel('Predicted', fontsize=9)
            plt.ylabel('Actual', fontsize=9)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)
            
            st.pyplot(fig_cm)
            
        st.info("AUC Score: 0.9763 | Precision: 0.93 | Recall: 0.94")

    with tab3:
        st.write("이미지에서 추출된 20개 특징의 실제 값입니다.")
        # FEATURE_ORDER가 정의되어 에러가 사라집니다.
        df_display = pd.DataFrame(extracted_data, columns=FEATURE_ORDER)
        st.dataframe(df_display.T.rename(columns={0: "수치"}), use_container_width=True)