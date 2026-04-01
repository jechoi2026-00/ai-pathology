import streamlit as st
import joblib
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from skimage.feature import graycomatrix, graycoprops

# --- [설정] 페이지 레이아웃 및 제목 (최상단 배치 필수) ---
st.set_page_config(page_title="암 진단 AI 리포트", layout="wide")

# --- 0. 전역 변수 설정 (모델이 기대하는 정확한 피처 순서) ---
FEATURE_ORDER = [
    'H_mean', 'S_mean', 'V_mean', 'G_mean', 'B_mean', 
    'Color_Variance', 'Contrast_0', 'Contrast_45', 'Contrast_135', 
    'Homogeneity_45', 'Homogeneity_90', 'Homogeneity_135', 
    'Correlation_avg', 'Energy_avg', 'ASM_avg', 'Entropy_avg', 
    'Gray_Mean_Center', 'Texture_Complexity', 'Uniformity_Score', 'Edge_Density'
]

# --- 1. 모델 및 스케일러 로드 ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('finals_tuned_xgboost_20feats.pkl') 
        scaler = joblib.load('scaler_20feats.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"모델 파일을 찾을 수 없습니다: {e}")
        return None, None

model, scaler = load_assets()

# --- 2. 핵심 특징 추출 함수 (학습 환경과 100% 동기화) ---
def extract_20_features(img_array):
    # A. PIL(RGB) -> OpenCV(BGR) 변환 (학습 시 cv2.imread 로직 재현)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # B. 중앙 32x32 패치 추출 (학습 시 img[32:64, 32:64] 고정 좌표 재현)
    h, w, _ = img_bgr.shape
    if h >= 64 and w >= 64:
        center_bgr = img_bgr[32:64, 32:64]
    else:
        # 이미지가 64px 미만일 경우 중앙 크롭으로 대체
        center_bgr = img_bgr[max(0,h//2-16):h//2+16, max(0,w//2-16):w//2+16]
    
    # C. 전처리용 이미지 생성
    gray = cv2.cvtColor(center_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(center_bgr, cv2.COLOR_BGR2HSV)
    
    features = {}
    
    # [색상 특징 - BGR 기준]
    features['H_mean'] = np.mean(hsv[:,:,0])
    features['S_mean'] = np.mean(hsv[:,:,1])
    features['V_mean'] = np.mean(hsv[:,:,2])
    features['G_mean'] = np.mean(center_bgr[:,:,1]) # BGR 중 G
    features['B_mean'] = np.mean(center_bgr[:,:,0]) # BGR 중 B
    features['Color_Variance'] = np.var(center_bgr)
    
    # [질감 특징 (GLCM)] - 거리 1, 4개 각도
    glcm = graycomatrix(gray, distances=[1], 
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                        levels=256, symmetric=True, normed=True)
    
    # 특정 각도별 특징 (Contrast & Homogeneity)
    features['Contrast_0'] = graycoprops(glcm, 'contrast')[0, 0]
    features['Contrast_45'] = graycoprops(glcm, 'contrast')[0, 1]
    features['Contrast_135'] = graycoprops(glcm, 'contrast')[0, 3]
    features['Homogeneity_45'] = graycoprops(glcm, 'homogeneity')[0, 1]
    features['Homogeneity_90'] = graycoprops(glcm, 'homogeneity')[0, 2]
    features['Homogeneity_135'] = graycoprops(glcm, 'homogeneity')[0, 3]
    
    # 평균 질감 특징
    features['Correlation_avg'] = np.mean(graycoprops(glcm, 'correlation'))
    features['Energy_avg'] = np.mean(graycoprops(glcm, 'energy'))
    features['ASM_avg'] = np.mean(graycoprops(glcm, 'ASM'))
    features['Entropy_avg'] = -np.sum(glcm * np.log2(glcm + 1e-10))
    
    # [추가 특징]
    features['Gray_Mean_Center'] = np.mean(gray) 
    features['Texture_Complexity'] = np.std(graycoprops(glcm, 'contrast'))
    features['Uniformity_Score'] = np.mean(graycoprops(glcm, 'homogeneity'))
    features['Edge_Density'] = np.mean(cv2.Canny(gray, 100, 200)) / 255.0

    # D. FEATURE_ORDER에 맞춰 리스트 생성
    ordered_values = [features.get(name, 0.0) for name in FEATURE_ORDER]
    return np.array(ordered_values).reshape(1, -1)

# --- 3. 메인 UI 화면 구성 ---
st.title("🔬 암 진단 AI 병리 리포트")
st.info("이미지를 업로드하면 AI가 [32:64, 32:64] 영역을 분석하여 암 조직 여부를 판독합니다.")

uploaded_file = st.file_uploader("조직 이미지(TIF/PNG/JPG) 업로드", type=['tif', 'tiff', 'png', 'jpg'])

if uploaded_file and model is not None:
    # 이미지 로딩 및 RGB 보장
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    
    # 상단 결과 레이아웃
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.subheader("🖼️ 분석 이미지")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # 크롭 영역 시각화
        h, w = img_array.shape[:2]
        if h >= 64 and w >= 64:
            img_viz = img_array.copy()
            cv2.rectangle(img_viz, (32, 32), (64, 64), (255, 0, 0), 2)
            st.image(img_viz, caption="🔴 분석 영역 표시 [32:64, 32:64]", use_container_width=True)

    with col2:
        # 예측 수행
        with st.spinner('분석 중...'):
            extracted_data = extract_20_features(img_array)
            scaled_data = scaler.transform(extracted_data)
            prediction = model.predict(scaled_data)[0]
            prob = model.predict_proba(scaled_data)[0][1]

        st.subheader("📋 판독 결과")
        if prediction == 1:
            st.error(f"### 🚨 암 조직 가능성 높음 ({prob*100:.2f}%)")
            st.progress(float(prob))
            st.warning("주의: 본 결과는 AI 보조 지표이며, 전문의의 최종 진단이 필요합니다.")
        else:
            st.success(f"### ✅ 정상 조직 가능성 높음 ({(1-prob)*100:.2f}%)")
            st.progress(float(1-prob))
            st.write("조직의 특징이 정상 범주 내에 있습니다.")

    st.divider()
    
    # --- 4. 하단 상세 분석 (Explainability) ---
    st.header("📊 AI 판단 근거 (Model Explainability)")
    
    tab1, tab2, tab3 = st.tabs(["✨ 특징 중요도", "🧪 모델 신뢰도", "📝 상세 수치"])
    
    with tab1:
        st.write("모델이 판독 시 가장 중요하게 고려한 TOP 10 요인입니다.")
        importances = model.feature_importances_
        fi_df = pd.DataFrame({'Feature': FEATURE_ORDER, 'Importance': importances})
        fi_df = fi_df.sort_values('Importance', ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=fi_df, x='Importance', y='Feature', palette='viridis', ax=ax)
        plt.title("Feature Importance (Top 10)")
        st.pyplot(fig)

    with tab2:
        st.write("학습 시 검증 데이터를 통한 모델의 정확도 지표입니다.")
        c1, c2, c3 = st.columns([1, 1.5, 1])
        with c2:
            cm_data = [[19463, 1537], [1269, 19731]] 
            fig_cm, ax_cm = plt.subplots(figsize=(4, 3)) 
            sns.heatmap(cm_data, annot=True, fmt='d', cmap='RdPu', 
                        xticklabels=['Normal', 'Cancer'], yticklabels=['Normal', 'Cancer'],
                        annot_kws={"size": 10})
            plt.xlabel('Predicted', fontsize=9)
            plt.ylabel('Actual', fontsize=9)
            st.pyplot(fig_cm)
        st.info("AUC Score: 0.9763 | Precision: 0.93 | Recall: 0.94")

    with tab3:
        st.write("이미지 [32:64, 32:64] 영역에서 추출된 20개 특징의 값입니다.")
        df_display = pd.DataFrame(extracted_data, columns=FEATURE_ORDER)
        st.dataframe(df_display.T.rename(columns={0: "수치"}), use_container_width=True)

else:
    if model is None:
        st.warning("모델 파일(.pkl)이 로드되지 않았습니다. 파일명을 확인해 주세요.")
