import streamlit as st

import cv2

import numpy as np

import joblib

import plotly.graph_objects as go

from skimage.feature import local_binary_pattern, graycomatrix, graycoprops



# --- 1. 특징 추출 함수 (원본 로직) ---

def get_32_features(patch):

    f = []

    # 색상 (RGB, HSV, LAB)

    for space in [None, cv2.COLOR_BGR2HSV, cv2.COLOR_BGR2LAB]:

        target = patch if space is None else cv2.cvtColor(patch, space)

        f.extend(np.mean(target, axis=(0,1)).tolist())

        f.extend(np.std(target, axis=(0,1)).tolist())

    # 질감 (GLCM)

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

    for prop in ['contrast', 'homogeneity', 'energy', 'correlation']:

        f.append(graycoprops(glcm, prop)[0, 0])

    # 패턴 (LBP)

    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')

    hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0,10), density=True)

    f.extend(hist.tolist())

    return np.array(f, dtype=np.float32)



def extract_logic_96x96(img_bgr):

    tile_features = []

    for y in range(0, 96, 32):

        for x in range(0, 96, 32):

            tile = img_bgr[y:y+32, x:x+32]

            if np.mean(tile) > 240: continue 

            tile_features.append(get_32_features(tile))

    if not tile_features:

        return get_32_features(img_bgr[32:64, 32:64])

    return np.max(np.array(tile_features), axis=0)



# --- 2. 모델 및 자산 로드 ---

@st.cache_resource

def load_assets():

    model = joblib.load('final_rf_model.pkl')

    scaler = joblib.load('scaler.pkl')

    selected_indices = joblib.load('selected_features.pkl') # [9, 17, 18, 20, 26]

    

    # 특징 이름 매핑 (사용자 로직 순서 기준)

    feature_names_all = [

        "R_mean", "G_mean", "B_mean", "R_std", "G_std", "B_std",

        "H_mean", "S_mean", "V_mean", "H_std", "S_std", "V_std",

        "L_mean", "A_mean", "B_mean", "L_std", "A_std", "B_std",

        "GLCM_Contrast", "GLCM_Homogeneity", "GLCM_Energy", "GLCM_Correlation",

        "LBP_0", "LBP_1", "LBP_2", "LBP_3", "LBP_4", "LBP_5", "LBP_6", "LBP_7", "LBP_8", "LBP_9"

    ]

    return model, scaler, selected_indices, feature_names_all



# --- 3. UI 구성 ---

st.set_page_config(page_title="Cancer AI Analysis", page_icon="🔬", layout="wide")



st.title("🔬 암 조직 병리 슬라이드 정밀 분석")

st.markdown("---")



try:

    model, scaler, selected_indices, feature_names_all = load_assets()

    st.sidebar.success("✅ AI 엔진 로드 완료")

except Exception as e:

    st.sidebar.error(f"❌ 파일 로드 실패: {e}")

    st.stop()



uploaded_file = st.file_uploader("96x96 조직 이미지를 업로드하세요.", type=['tif', 'png', 'jpg'])



if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    

    col1, col2 = st.columns([1, 1.5])

    

    with col1:

        st.image(img_rgb, caption="업로드 이미지", use_container_width=True)

        

        # 특징 추출 및 예측

        features_32 = extract_logic_96x96(img_bgr)

        features_5 = features_32[selected_indices].reshape(1, -1)

        features_scaled = scaler.transform(features_5)[0]

        prob = model.predict_proba(features_5)[0][1] # 스케일링 전/후는 모델 설정에 맞게 확인

        threshold = 0.4380

        

    with col2:

        # 가우지 차트 (확률 시각화)

        fig_gauge = go.Figure(go.Indicator(

            mode = "gauge+number",

            value = prob * 100,

            domain = {'x': [0, 1], 'y': [0, 1]},

            title = {'text': "암 발생 위험도 (%)", 'font': {'size': 24}},

            gauge = {

                'axis': {'range': [None, 100], 'tickwidth': 1},

                'bar': {'color': "darkblue"},

                'bgcolor': "white",

                'borderwidth': 2,

                'bordercolor': "gray",

                'steps': [

                    {'range': [0, threshold*100], 'color': 'lightgreen'},

                    {'range': [threshold*100, 100], 'color': 'salmon'}],

                'threshold': {

                    'line': {'color': "red", 'width': 4},

                    'thickness': 0.75,

                    'value': threshold * 100}

            }

        ))

        fig_gauge.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))

        st.plotly_chart(fig_gauge, use_container_width=True)



        if prob > threshold:

            st.error(f"### 🚨 판독 결과: 암 조직 의심 (위험군)")

        else:

            st.success(f"### ✅ 판독 결과: 정상 조직 가능성 높음")



    st.markdown("---")

    

    # 하단 상세 그래프 영역

    st.subheader("📊 핵심 특징별 수치 분석")

    

    # 막대 그래프 데이터 준비

    selected_names = [feature_names_all[i] for i in selected_indices]

    

    fig_bar = go.Figure(data=[

        go.Bar(name='추출된 수치', x=selected_names, y=features_scaled, marker_color='teal')

    ])

    fig_bar.update_layout(

        title="선택된 특징의 스케일링된 수치 (VIF 최적화 결과)",

        xaxis_title="핵심 특징",

        yaxis_title="Scaled Value",

        height=400

    )

    st.plotly_chart(fig_bar, use_container_width=True)



    with st.expander("💡 그래프 해석 가이드"):

        st.write(f"1. **위험도 게이지**: 빨간색 경계선({threshold*100}%)을 넘으면 암으로 분류합니다.")

        st.write(f"2. **특징 수치**: 현재 이미지에서 추출된 {selected_names} 값들이 모델 판단의 근거가 됩니다.")

        st.write(f"3. **현재 확률값**: {prob:.4f} (임계값 대비 매우 {'낮음' if prob < threshold else '높음'})")

    with st.expander("💡 그래프 및 특징 상세 설명"):

        st.write("AI가 암을 판독할 때 가장 중요하게 본 5가지 단서입니다:")

        st.write("- **H_std**: 색상의 다양성 (암세포 핵의 불규칙한 염색 정도)")

        st.write("- **B_std**: 색 농도의 변화량 (조직 내 염색의 불균형함)")

        st.write("- **Contrast**: 조직의 거친 정도 (인접 세포 간의 급격한 변화)")

        st.write("- **Energy**: 질감의 규칙성 (정상의 매끄러움 vs 암의 복잡함)")

        st.write("- **LBP_4**: 미세 형태 패턴 (암세포 특유의 기하학적 지문)")

        st.info(f"현재 업로드된 이미지의 예측 확률은 **{prob:.4f}**이며, 기준점({threshold}) 대비 분석된 결과입니다.")

st.caption("⚠️ 본 시스템은 연구용 모델입니다.")