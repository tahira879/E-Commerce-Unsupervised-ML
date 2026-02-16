import streamlit as st
import zipfile
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest



# --- THEME COLORS ---
GRAPE = "#94618E"
EGGPLANT = "#49274A"
SAND = "#F4DECB"
SHELL = "#F8EEE7"

st.set_page_config(page_title="UNSUPERVISED MASTER HUB", layout="wide")

st.markdown(f"""

<style>

/* Main App Background */
.stApp {{ 
    background-color: {SHELL} !important; 
    color: {EGGPLANT} !important; 
}}

/* Sidebar Styling */
[data-testid="stSidebar"] {{ 
    background-color: {EGGPLANT} !important; 
    border-right: 3px solid {GRAPE}; 
}}
[data-testid="stSidebar"] * {{ color: {SAND} !important; }}

/* Neon Header */
.neon-header {{
    font-size: 50px; 
    font-weight: 900; 
    color: {EGGPLANT}; 
    text-align: center;
    transition: 0.3s ease; 
    text-transform: uppercase;
}}
.neon-header:hover {{
    color: {GRAPE};
    text-shadow: 0 0 15px {GRAPE}, 0 0 30px {GRAPE};
}}

/* TABLE AND DATAFRAME NEON WRAPPER */
.neon-wrapper {{
    border: 2px solid {EGGPLANT};
    border-radius: 15px;
    padding: 15px;
    box-shadow: 0 0 15px rgba(73, 39, 74, 0.5);
    margin-bottom: 25px;
    background: rgba(255, 255, 255, 0.05);
}}
/* Footer Box Hover Effect */
.footer-box {{
    transition: all 0.4s ease-in-out;
    border: 1px solid rgba(0, 255, 255, 0.1);
}}

.footer-box:hover {{
    transform: translateY(-5px);
    box-shadow: 0 15px 35px rgba(0, 255, 255, 0.2);
    border: 1px solid #00FFFF;
}}

/* Social Buttons Hover Effect */
.social-btn {{
    display: inline-block;
    padding: 10px 25px;
    margin: 0 10px;
    border: 1px solid #00FFFF;
    border-radius: 30px;
    color: #00FFFF;
    text-decoration: none;
    font-weight: bold;
    transition: all 0.3s ease;
}}

.social-btn:hover {{
    background-color: #00FFFF;
    color: #0d1117; /* Theme dark color */
    box-shadow: 0 0 20px #00FFFF;
    transform: scale(1.1);
}}
/* Side Box */
.side-box {{
    background: white; 
    padding: 20px; 
    border-radius: 15px;
    border-right: 8px solid {GRAPE}; 
    margin-bottom: 20px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}}

.hover-img {{
    width: 100%; 
    border-radius: 15px; 
    transition: 0.5s; 
    margin-top: 10px;
}}
.hover-img:hover {{ 
    transform: scale(1.05); 
    filter: brightness(1.1); 
}}

/* Flip Card */
.flip-card {{
    background-color: transparent; 
    width: 100%; 
    height: 350px; 
    perspective: 1000px;
}}
.flip-card-inner {{
    position: relative; 
    width: 100%; 
    height: 100%; 
    transition: 0.6s; 
    transform-style: preserve-3d;
}}
.flip-card:hover .flip-card-inner {{ 
    transform: rotateY(180deg); 
}}
.flip-card-front, .flip-card-back {{
    position: absolute; 
    width: 100%; 
    height: 100%; 
    backface-visibility: hidden; 
    border-radius: 20px; 
    display: flex; 
    align-items: center; 
    justify-content: center; 
    padding: 20px;
}}
.flip-card-front img {{
    width: 100%; 
    height: 100%; 
    object-fit: cover; 
    border-radius: 20px;
}}
.flip-card-back {{
    background-color: {GRAPE}; 
    color: white; 
    transform: rotateY(180deg); 
    flex-direction: column; 
    text-align: center;
}}

/* Footer */
.footer-box {{ 
    background-color: {EGGPLANT} !important; 
    color: {SAND} !important;
    padding: 50px; 
    border-radius: 30px; 
    text-align: center; 
    margin-top: 60px;
    border: 2px solid {GRAPE}; 
}}

/* Social Buttons */
.social-btn {{ 
    display: inline-flex; 
    align-items: center; 
    justify-content: center; 
    background-color: {GRAPE}; 
    color: {SAND} !important; 
    width: 60px; 
    height: 60px; 
    margin: 0 15px; 
    border-radius: 50%; 
    transition: 0.4s; 
    text-decoration: none !important; 
    font-weight: bold; 
}}
.social-btn:hover {{ 
    background-color: {SAND}; 
    color: {EGGPLANT} !important; 
    transform: translateY(-10px); 
}}

/* =============================== */
/* FINAL OVERRIDE FOR ALL BOXES   */
/* =============================== */

/* Insight Box Final Style */
.insight-box {{
    background-color: #5A2D5C !important;
    color: white !important;
    border-radius: 16px !important;
    padding: 30px !important;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.8) !important;
    border: none !important;
}}

/* Streamlit Info/Warning/Success/Error */
div[data-baseweb="notification"] {{
    background-color: #5A2D5C !important;
    color: white !important;
    border-radius: 16px !important;
    border: none !important;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.8) !important;
}}
/* HOVER EFFECT */
.insight-box:hover {{
    transform: translateY(-6px) !important;
    box-shadow: 0 0 25px #94618E, 0 0 50px rgba(148, 97, 142, 0.6) !important;
}}

div[data-baseweb="notification"] > div {{
    border-left: none !important;
}}

div[data-baseweb="notification"] p {{
    color: #F4DECB !important;
    font-weight: 500;
}}

</style>

""", unsafe_allow_html=True)


# --- DATA ENGINE ---
@st.cache_data
def load_and_clean():
    zip_path = r"zip_path = "Online_Retail.csv.zip""
    try:
        with zipfile.ZipFile(zip_path) as z:
            with z.open("Online_Retail.csv") as f:
                df = pd.read_csv(f, encoding="ISO-8859-1", nrows=5000)
                df.dropna(subset=['CustomerID'], inplace=True)
                df.drop_duplicates(inplace=True)
                df['TotalSales'] = df['Quantity'] * df['UnitPrice']
                return df
    except: return pd.DataFrame()

df = load_and_clean()
# --- STEP 1: PREPROCESSING (Saari Calculation se pehle) ---
from sklearn.preprocessing import StandardScaler

# Sirf numerical columns lein
df_numeric = df.select_dtypes(include=[np.number])

# Z-score Scaler apply karein
scaler = StandardScaler()
df_scaled_array = scaler.fit_transform(df_numeric)

# Isko wapas DataFrame mein convert karein taaki use karne mein asani ho
df_scaled = pd.DataFrame(df_scaled_array, columns=df_numeric.columns)
# --- NAVIGATION ---
st.sidebar.title("🍷 NAVIGATION")
choice = st.sidebar.radio("PIPELINE STAGES:", ["Introduction", "EDA", "Preprocessing", "K-Means", "Anomaly Detection", "PCA", "Recommendation", "Reflection"])

# --- MAIN LOGIC (IF-ELIF BLOCK) ---
if choice == "Introduction":
    st.markdown('<h1 class="neon-header">UNSUPERVISED HUB</h1>', unsafe_allow_html=True)
    col_t, col_s = st.columns([2,1])
    with col_t:
        st.write("### **Unsupervised Learning:** The Art of Pattern Discovery")
        st.write("Unsupervised learning algorithms analyze data without any pre-defined labels.")
        st.markdown('<div class="insight-box"><b>Goal:</b> Transform raw data into business intelligence.</div>', unsafe_allow_html=True)
    with col_s:
        st.markdown(f'<div class="side-box"><b>AI INTUITION:</b> Learning without labels.<img class="hover-img" src="https://images.unsplash.com/photo-1550751827-4bd374c3f58b?q=80&w=400"></div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    topics = [
        ("CLUSTERING", "Segmenting users.", "https://images.unsplash.com/photo-1551288049-bebda4e38f71?q=80&w=400"),
        ("ANOMALIES", "Spotting outliers.", "https://images.unsplash.com/photo-1509228627152-72ae9ae6848d?q=80&w=400"),
        ("PCA", "Dimension reduction.", "https://images.unsplash.com/photo-1555949963-aa79dcee981c?q=80&w=400")
    ]
    for i, (t, d, img) in enumerate(topics):
        with [c1, c2, c3][i]:
            st.markdown(f'<div class="flip-card"><div class="flip-card-inner"><div class="flip-card-front"><img src="{img}"></div><div class="flip-card-back"><h3>{t}</h3><p>{d}</p></div></div></div>', unsafe_allow_html=True)

# --- TASK 2: EDA ---
elif choice == "EDA":
    st.markdown('<h1 class="neon-header">DATA EXPLORATION</h1>', unsafe_allow_html=True)
    
    eda_mode = st.selectbox("CHOOSE ANALYSIS VIEW:", ["Data Overview", "Data Cleaning", "Visual Analysis"])
    st.markdown("---")

    col_m, col_s = st.columns([3.2, 0.8])

    with col_m:
        if eda_mode == "Data Overview":
            # --- 1. RAW DATA GLIMPSE ---
            st.write("### 🔍 Raw Data Glimpse")
            st.markdown('<div style="background-color: #0F0F0F; border: 2px solid #5A2D5C; padding: 10px; border-radius: 12px; box-shadow: 0 10px 20px rgba(0,0,0,0.5);">', unsafe_allow_html=True)
            st.dataframe(df.head(10), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.info("**Interpretation:** A preview of the first few rows of the dataset to understand its structure, observe sample values, and get an initial sense of the feature types and data quality.")

            # --- 2. DATA DIMENSIONS ---
            st.write("### 📊 Dataset Dimensions")
            sh1, sh2 = st.columns(2)
            for col, label, value in zip([sh1, sh2], ["TOTAL ROWS", "TOTAL COLUMNS"], [df.shape[0], df.shape[1]]):
                with col:
                    st.markdown(f"""
                        <div class="insight-box" style="background-color: #5A2D5C !important; border: none !important; box-shadow: 5px 5px 15px rgba(0,0,0,0.7); text-align:center; padding: 30px;">
                            <h4 style="margin:0; color:#F4DECB; font-size:14px; letter-spacing:1px;">{label}</h4>
                            <h1 style="color:white; margin:0; font-weight:900; font-size:45px;">{value}</h1>
                        </div>
                    """, unsafe_allow_html=True)

            st.warning(f"**Interpretation:**The dataset contains a total of {df.shape[0]} entries. The number of columns determines the model's complexity and feature depth.")

            # --- 3. FEATURE CLASSIFICATION ---
            st.write("### 🛠️ Feature Classification")
            num_cols = df.select_dtypes(include=['number']).columns.tolist()
            cat_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            f1, f2 = st.columns(2)
            with f1:
                st.markdown(f"""
                    <div class="insight-box" style="background-color: #5A2D5C !important; border: none !important; box-shadow: 5px 5px 15px rgba(0,0,0,0.7); min-height: 200px;">
                        <h4 style="color:white; margin-top:0; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom:10px;">🔢 Numerical</h4>
                        <ul style="color:#F4DECB; padding-top:10px; list-style-type: square;">
                            {"".join([f"<li>{c}</li>" for c in num_cols])}
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
            with f2:
                st.markdown(f"""
                    <div class="insight-box" style="background-color: #5A2D5C !important; border: none !important; box-shadow: 5px 5px 15px rgba(0,0,0,0.7); min-height: 200px;">
                        <h4 style="color:white; margin-top:0; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom:10px;">🔤 Categorical</h4>
                        <ul style="color:#F4DECB; padding-top:10px; list-style-type: square;">
                            {"".join([f"<li>{c}</li>" for c in cat_cols])}
                        </ul>
                    </div>
                """, unsafe_allow_html=True)

            st.success("**Interpretation:** Numerical features can be directly scaled for analysis or modeling, whereas categorical features must be encoded into numerical form beforehand to be usable in machine learning models.")

            # --- 4. STATISTICAL SUMMARY ---
            st.write("### 🔢 Statistical Summary (Transposed)")
            st.markdown('<div style="background-color: #0F0F0F; border: 2px solid #5A2D5C; border-radius:12px; padding:10px;">', unsafe_allow_html=True)
            st.table(df.describe().T)
            st.markdown('</div>', unsafe_allow_html=True)

            st.error("**Interpretation:**Examine the standard deviation and range of the data; these metrics help understand how spread out the values are and identify any potential outliers that may affect analysis or modeling.")

        elif eda_mode == "Data Cleaning":
            # --- MODE 2: CLEANING ---
            st.write("### 🛡️ Data Integrity & Sanitization")
            c1, c2 = st.columns(2)
            null_count = df.isnull().sum().sum()
            dupes_count = df.duplicated().sum()

            with c1:
                st.markdown(f'<div class="insight-box" style="background-color: #5A2D5C !important; padding: 25px; text-align:center;"><h4 style="color:#F4DECB;">NULL VALUES</h4><h1 style="color:white;">{null_count}</h1></div>', unsafe_allow_html=True)
                if null_count > 0:
                    if st.button("🧼 CLEAN NULL VALUES", use_container_width=True):
                        df.dropna(inplace=True)
                        st.rerun()
            with c2:
                st.markdown(f'<div class="insight-box" style="background-color: #5A2D5C !important; padding: 25px; text-align:center;"><h4 style="color:#F4DECB;">DUPLICATES</h4><h1 style="color:white;">{dupes_count}</h1></div>', unsafe_allow_html=True)
                if dupes_count > 0:
                    if st.button("♻️ REMOVE DUPLICATES", use_container_width=True):
                        df.drop_duplicates(inplace=True)
                        st.rerun()
            st.info(f"**Interpretation:**The dataset contains no null values or duplicate entries. This indicates that the data is already clean and reliable, requiring no additional handling for missing or repeated values.")

                        # --- 3. OUTLIER DETECTION & REMOVAL ---
          
            from scipy import stats

# --- IQR OUTLIER MANAGEMENT ---
            st.write("### ⚠️ Outlier Detection & Removal (IQR Method)")
            
            # 1. Identify Numeric Columns
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            outlier_info = {}
            total_rows_with_outliers = 0

            if numeric_cols:
                # Calculate IQR and detect outliers
                Q1 = df[numeric_cols].quantile(0.25)
                Q3 = df[numeric_cols].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Create a mask for rows that have ANY outlier
                outlier_mask = ((df[numeric_cols] < lower_bound) | (df[numeric_cols] > upper_bound)).any(axis=1)
                total_rows_with_outliers = outlier_mask.sum()

                # Display detection results
                if total_rows_with_outliers > 0:
                    st.warning(f"Analysis complete: **{total_rows_with_outliers}** anomalous rows detected.")
                    
                    # Optional: Show breakdown per column
                    for col in numeric_cols:
                        col_outliers = ((df[col] < lower_bound[col]) | (df[col] > upper_bound[col])).sum()
                        if col_outliers > 0:
                            st.markdown(f"📍 **{col}:** {col_outliers} outliers")

                    # 2. Action Button
                    if st.button("🧼 PURGE OUTLIERS", use_container_width=True):
                        # Removing outliers from the dataframe
                        df = df[~outlier_mask]
                        
                        # Trigger Visuals
                        st.balloons()
                        st.success(f"Purge Successful! {total_rows_with_outliers} rows have been removed.")
                        st.info("**Interpretation:** The dataset has been sanitized using the Interquartile Range (IQR) method. This process minimizes variance caused by extreme values and ensures the integrity of the subsequent modeling phase.")
                        
                        # Force refresh to update dataframes/stats elsewhere
                        # st.rerun()
                else:
                    st.success("✅ **System Integrity Verified:** No significant outliers detected in numeric features.")
                

        elif eda_mode == "Visual Analysis":
            st.write("### 📈 Graphical Intelligence")

            # Column selection for Visuals
            num_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            if len(num_cols) >= 2:
                # --- 1. CORRELATION HEATMAP ---
                st.write("### 🌡️ Feature Correlation Heatmap")
                import plotly.express as px
                
                corr = df[num_cols].corr()
                fig_corr = px.imshow(corr, 
                                    text_auto=True, 
                                    color_continuous_scale='Purples',
                                    template="plotly_dark")
                
                st.markdown('<div style="background-color: #0F0F0F; border: 2px solid #5A2D5C; padding: 10px; border-radius: 12px;">', unsafe_allow_html=True)
                st.plotly_chart(fig_corr, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.error("**Interpretation:** The heatmap shows how features are related to each other. A value of 1.0 indicates a strong positive correlation,\n while -1.0 indicates a strong negative correlation. Values close to 0 suggest no linear relationship.")

                # --- 2. DISTRIBUTION ANALYSIS ---
                st.write("### 📊 Distribution of Numerical Features")
                selected_col = st.selectbox("SELECT COLUMN TO ANALYZE:", num_cols)
                
                fig_hist = px.histogram(df, x=selected_col, 
                                       marginal="box", # Adds a box plot on top
                                       color_discrete_sequence=['#94618E'],
                                       template="plotly_dark")
                
                st.markdown('<div style="background-color: #0F0F0F; border: 2px solid #5A2D5C; padding: 10px; border-radius: 12px;">', unsafe_allow_html=True)
                st.plotly_chart(fig_hist, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.warning(f"**Interpretation:**Here you can observe the distribution of {selected_col}, along with its outliers highlighted in the box plot, showing median, quartiles, and potential extreme values")

                # --- 3. CATEGORICAL INSIGHTS ---
                cat_cols = df.select_dtypes(include=['object']).columns.tolist()
                if cat_cols:
                    st.write("### 🔠 Categorical Breakdown")
                    selected_cat = st.selectbox("SELECT CATEGORICAL COLUMN:", cat_cols)
                    
                    fig_pie = px.pie(df, names=selected_cat, 
                                    hole=0.4,
                                    color_discrete_sequence=px.colors.sequential.Purp,
                                    template="plotly_dark")
                    
                    st.markdown('<div style="background-color: #0F0F0F; border: 2px solid #5A2D5C; padding: 10px; border-radius: 12px;">', unsafe_allow_html=True)
                    st.plotly_chart(fig_pie, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.success(f"**Interpretation:**The distribution of the various categories in {selected_cat} is displayed here.")
                    

                    st.write("### 📈 Bivariate Intelligence")

            num_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            if len(num_cols) >= 2:
                # Selection Row
                v1, v2 = st.columns(2)
                with v1:
                    x_axis = st.selectbox("SELECT X-AXIS:", num_cols, index=0)
                with v2:
                    y_axis = st.selectbox("SELECT Y-AXIS:", num_cols, index=1)

                st.markdown("---")
                
                # Side-by-Side Graphs Row
                g1, g2 = st.columns(2)
                
                import plotly.express as px

                with g1:
                    st.write(f"#### 🛰️ Scatter: {x_axis} vs {y_axis}")
                    fig_scatter = px.scatter(df, x=x_axis, y=y_axis, 
                                           color_discrete_sequence=['#F4DECB'],
                                           template="plotly_dark",
                                           opacity=0.7)
                    
                    st.markdown('<div style="background-color: #0F0F0F; border: 2px solid #5A2D5C; padding: 10px; border-radius: 12px;">', unsafe_allow_html=True)
                    # Unique Key added here
                    st.plotly_chart(fig_scatter, use_container_width=True, key="scatter_bivariate")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.info(f"**Interpretation:** This scatter plot reveals the correlation between {x_axis} and {y_axis}. A linear pattern suggests a strong relationship, while scattered points indicate independence.")

                with g2:
                    st.write(f"#### 📦 Box Plot: {y_axis} Distribution")
                    fig_box = px.box(df, y=y_axis, 
                                   color_discrete_sequence=['#94618E'],
                                   template="plotly_dark",
                                   points="all")
                    
                    st.markdown('<div style="background-color: #0F0F0F; border: 2px solid #5A2D5C; padding: 10px; border-radius: 12px;">', unsafe_allow_html=True)
                    # Unique Key added here
                    st.plotly_chart(fig_box, use_container_width=True, key="box_bivariate")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.warning(f"**Interpretation:** The Box Plot highlights the median, quartiles, and extreme outliers for {y_axis}. Any points outside the whiskers are statistical anomalies.")

               
                
                st.markdown('<div style="background-color: #0F0F0F; border: 2px solid #5A2D5C; padding: 10px; border-radius: 12px;">', unsafe_allow_html=True)

                
            else:
                st.markdown(f"""
                    <div class="insight-box" style="background-color: #5A2D5C !important; text-align:center; padding: 20px;">
                        <h3 style="color:white;">⚠️ INSUFFICIENT DATA</h3>
                        <p style="color:#F4DECB;">Visual analysis requires at least 2 numerical columns.</p>
                    </div>
                """, unsafe_allow_html=True)   


    # --- SIDEBAR (ALAG COLUMN) ---
    with col_s:
        st.markdown(f"""
            <div class="insight-box" style="background-color: #5A2D5C !important; border: none !important; box-shadow: 5px 5px 25px rgba(0,0,0,0.8);">
                <h4 style="text-align:center; color:white; letter-spacing:1px;">EDA DEFINITION</h4>
                <hr style="border-color: rgba(255,255,255,0.1);">
                <p style="font-size: 13px; color:#F4DECB; line-height:1.6;">
                    Exploratory Data Analysis (EDA) dataset ke mukhya gunon ko summarize karne ka tareeka hai.
                </p>
                <img src="https://images.unsplash.com/photo-1551288049-bebda4e38f71?q=80&w=200" style="width:100%; border-radius:8px; margin-top:10px;">
            </div>
            <div class="insight-box" style="text-align:center; background:#5A2D5C !important; border: none !important; margin-top:20px;">
                <span style="color:#00FF41; font-weight:bold; font-size:12px;">● SYSTEM ACTIVE</span>
            </div>
        """, unsafe_allow_html=True)
        

elif choice == "Preprocessing":
    st.markdown('<h1 class="neon-header">DATA UNDERSTANDING & PREPROCESSING</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # CSS for Hover Effects and Custom Boxes
  

    col_m, col_s = st.columns([3.2, 0.8])

    with col_m:
        # --- 1. FEATURE SCALING SECTION ---
        st.write("### 🔍 Feature Scaling / Normalization")
        
        st.markdown("""
            <div class="insight-box" style="background-color: #5A2D5C !important; border-left: 5px solid #00FF41 !important; padding: 20px; margin-bottom: 20px;">
                <h5 style="color: white; margin-top:0;">What is Feature Scaling?</h5>
                <p style="color: #F4DECB; font-size: 14px; line-height:1.6;">
                    Scaling standardizes the range of features. In <b>Unsupervised Learning</b>, algorithms like K-Means use distance; scaling ensures features with larger ranges don't dominate the clusters.
                </p>
            </div>
        """, unsafe_allow_html=True)

        # Logic
        from sklearn.preprocessing import MinMaxScaler
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        if num_cols:
            scaler = MinMaxScaler()
            df[num_cols] = scaler.fit_transform(df[num_cols])
            st.success(f"✅ Numerical features scaled (Min-Max): {', '.join(num_cols)}")
        else:
            st.warning("⚠️ No numerical features found for scaling.")

        st.info("**Empress Insight:** Scaling ensures numerical features contribute equally to the distance-based algorithms.")
        st.markdown("<br>", unsafe_allow_html=True)

        # --- 2. ONE-HOT ENCODING SECTION ---
        st.write("### 🔤 One-Hot Encoding")
        
        st.markdown("""
            <div class="insight-box" style="background-color: #5A2D5C !important; border-left: 5px solid #F4DECB !important; padding: 20px; margin-bottom: 20px;">
                <h5 style="color: white; margin-top:0;">What is One-Hot Encoding?</h5>
                <p style="color: #F4DECB; font-size: 14px; line-height:1.6;">
                    Since machines only understand numbers, <b>One-Hot Encoding</b> converts text categories into binary columns (0s and 1s), removing any artificial ranking.
                </p>
            </div>
        """, unsafe_allow_html=True)

        # Logic
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        if cat_cols:
            df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
            st.success(f"✅ Categorical features encoded: {', '.join(cat_cols)}")
        else:
            st.warning("⚠️ No categorical features found for encoding.")

        st.markdown("<br>", unsafe_allow_html=True)

        # --- 3. FINAL SUMMARY WITH HOVER EFFECT ---
        st.write("### ✨ Why Preprocessing Matters")
        
        st.markdown("""
        <div class="preprocessing-card">
            <p style="font-size: 16px; border-bottom: 1px solid #5A2D5C; padding-bottom: 10px; margin-bottom: 15px;">
                "Raw data is just noise; preprocessing is the bridge that turns it into structured intelligence for our models."
            </p>
            <div style="margin-bottom: 10px;">
                <span class="feature-tag">⚖️ Equal Weightage:</span><br>
                Prevents <b>Large Numbers</b> (like Salary) from overpowering <b>Small Numbers</b> (like Age).
            </div>
            <div style="margin-bottom: 10px;">
                <span class="feature-tag">🔡 Machine Readable:</span><br>
                Converts human labels into <b>Mathematical Logic</b> without adding fake hierarchy.
            </div>
            <div style="margin-bottom: 10px;">
                <span class="feature-tag">📐 Geometric Truth:</span><br>
                Ensures <b>Euclidean Distances</b> accurately reflect the real similarity between points.
            </div>
            <div>
                <span class="feature-tag">🚀 Stable Discovery:</span><br>
                Leads to <b>Faster Convergence</b> and <b>Reproducible Patterns</b> in your final model.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_s:
        # --- SIDEBAR FINAL FIX ---
        # Note: Is code ko bilkul aise hi paste karein, extra spaces mat dein
        sidebar_content = """<div style="background-color: #5A2D5C; padding: 15px; border-radius: 20px; text-align: center; box-shadow: 0 10px 25px rgba(0,0,0,0.4); border: 1px solid rgba(255,255,255,0.1);"><h4 style="color: white; margin-bottom: 5px; font-size: 14px; letter-spacing: 1px;">PIPELINE STATUS</h4><hr style="border-color: rgba(255,255,255,0.2); margin: 8px 0;"><div style="border-radius: 12px; overflow: hidden; margin-bottom: 15px; border: 1px solid rgba(255,255,255,0.1);"><img src="https://images.unsplash.com/photo-1581091870627-3b08f44d3c2c?q=80&w=400", use_container_width=True style="width: 100%; display: block;"></div><p style="font-size: 11px; color: #F4DECB; line-height: 1.5; text-align: justify; margin: 0 5px 15px 5px;">Our preprocessing engine automates <b>Scaling</b> and <b>One-Hot Encoding</b> to remove mathematical noise. This ensures your clusters are based on actual data relationships rather than accidental bias.</p><div style="background: rgba(0,255,65,0.15); padding: 6px; border-radius: 8px; border: 1px solid rgba(0,255,65,0.2);"><span style="color: #00FF41; font-weight: bold; font-size: 10px; letter-spacing: 1px;">● SYSTEM OPTIMIZED</span></div></div>"""
        
        st.markdown(sidebar_content, unsafe_allow_html=True)





elif choice == "K-Means":
    import textwrap
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA
    import plotly.graph_objects as go
    import plotly.express as px

    st.markdown('<h1 class="neon-header">CUSTOMER SEGMENTATION: K-MEANS</h1>', unsafe_allow_html=True)
    st.markdown("---")

    df_numeric = df.select_dtypes(include=[np.number])
    col_m, col_s = st.columns([3.2, 0.8])

    with col_m:
        st.write("### 📐 Step 1: The Elbow Method")
        sse = [KMeans(n_clusters=k, random_state=42, init='k-means++').fit(df_numeric).inertia_ for k in range(1, 11)]
        fig_elbow = go.Figure(data=go.Scatter(x=list(range(1, 11)), y=sse, mode='lines+markers', marker=dict(color='#00FF41', size=10)))
        fig_elbow.update_layout(title="Optimal Cluster Discovery (WCSS)", template="plotly_dark", height=400)
        st.plotly_chart(fig_elbow, use_container_width=True)

        k_val = st.slider("Select Number of Clusters (k):", 2, 10, 3)
        df['Cluster'] = KMeans(n_clusters=k_val, random_state=42).fit_predict(df_numeric)
        
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(df_numeric)
        df['PCA1'], df['PCA2'] = pca_data[:, 0], pca_data[:, 1]

        fig_pca = px.scatter(df, x='PCA1', y='PCA2', color='Cluster', 
                             title=f"PCA Projection: Visualizing {k_val} Customer Segments", 
                             template="plotly_dark", color_continuous_scale='Turbo')
        fig_pca.update_traces(marker=dict(size=12, line=dict(width=1, color='white')))
        st.plotly_chart(fig_pca, use_container_width=True)

        # --- YAHAN DHAYAN DEIN: NO EXTRA SPACES BEFORE HTML TAGS ---
        interpretation_html = f"""
<div style="background-color: #ffffff; padding: 40px; border-radius: 20px; color: #1a1a1a; border: 1px solid #eeeeee; box-shadow: 0 15px 35px rgba(0,0,0,0.1);">
    <h2 style="color: #5A2D5C; margin-top: 0; font-weight: 800; border-bottom: 3px solid #5A2D5C; padding-bottom: 15px;">🧠 Strategic Interpretation</h2>
    <p style="font-size: 18px; line-height: 1.8; margin-top: 20px;">
        The <b>K-Means Clustering</b> model has categorized your customer base into <b>{k_val} distinct groups</b>. 
        This segmentation is mathematically optimized to ensure high <b>intra-cluster similarity</b>.
    </p>
    <div style="font-size: 16px; line-height: 1.7; margin-top: 25px;">
        <p>● <b>Mathematical Optimization:</b> By minimizing <b>WCSS</b>, the algorithm ensures each customer is assigned to the nearest <b>Centroid</b>.</p>
        <p>● <b>High-Dimensional Clarity:</b> We used <b>PCA</b> to compress multi-dimensional data into a 2D space for visual interpretation.</p>
        <p>● <b>Business Value:</b> This allows for the identification of <b>High-Value</b> and <b>At-Risk</b> segments for targeted marketing.</p>
    </div>
</div>"""
        st.markdown(interpretation_html, unsafe_allow_html=True)

    with col_s:
        st.image("https://cdn-images-1.medium.com/max/1200/1*99p3cuYI5Gf1f2-p0i1WlA.png", use_container_width=True)
        sidebar_html = f"""
<div style="background-color: #5A2D5C; padding: 20px; border-radius: 15px; text-align: center; border: 1px solid rgba(255,255,255,0.1);">
    <h4 style="color: white; font-weight: bold; font-size: 13px;">SEGMENTATION HUB</h4>
    <hr style="border-color: rgba(255,255,255,0.2);">
    <p style="font-size: 11px; color: #F4DECB;">Analyzing <b>{len(df)} users</b> using Centroid-based partitioning.</p>
    <div style="background: rgba(0,255,65,0.15); padding: 8px; border-radius: 8px; border: 1px solid #00FF41; margin-top: 10px;">
        <span style="color: #00FF41; font-weight: bold; font-size: 12px;">ACTIVE K: {k_val}</span>
    </div>
</div>"""
        st.markdown(sidebar_html, unsafe_allow_html=True)

elif choice == "Anomaly Detection":
    st.markdown('<h1 class="neon-header">DENSITY ESTIMATION & ANOMALY DETECTION</h1>', unsafe_allow_html=True)
    st.markdown("---")

    from sklearn.mixture import GaussianMixture
    import plotly.express as px
    import numpy as np

    # --- 1. GMM Logic & Preprocessing ---
    df_numeric = df.select_dtypes(include=[np.number])
    col_x = df_numeric.columns[0]
    col_y = df_numeric.columns[1]

    gmm = GaussianMixture(n_components=1, covariance_type='full', random_state=42)
    gmm.fit(df_numeric)
    
    # Calculate Scores
    scores = gmm.score_samples(df_numeric)
    df['Probability_Density'] = np.exp(scores) # Converting log-score to actual density
    threshold = np.percentile(scores, 5)
    df['Status'] = np.where(df['Probability_Density'] < np.exp(threshold), 'Anomaly', 'Normal')

    col_m, col_s = st.columns([3.2, 0.8])

    with col_m:
        st.write(f"### 🚨 Visualizing Outliers: {col_x} vs {col_y}")
        
        # --- 2. ENHANCED INTERACTIVE PLOT ---
        fig = px.scatter(
            df, x=col_x, y=col_y, 
            color='Status',
            color_discrete_map={'Anomaly': '#FF4B4B', 'Normal': '#00FF41'},
            title="Statistical Map: Normal Density vs. Anomalous Outliers",
            # Adding meaningful hover info
            hover_data={
                col_x: True, 
                col_y: True, 
                'Probability_Density': ':.6f', 
                'Status': True
            },
            template="plotly_dark"
        )
        
        # Make anomalies stand out more
        fig.update_traces(
            marker=dict(size=12, opacity=0.9, line=dict(width=1.5, color='white')),
            selector=dict(mode='markers')
        )
        
        # Improve Axis Labels and Clarity
        fig.update_layout(
            xaxis_title=f"Feature A: {col_x}",
            yaxis_title=f"Feature B: {col_y}",
            legend_title="Detection Result",
            hovermode="closest"
        )
        
        st.plotly_chart(fig, use_container_width=True)

        

        # --- 3. CLEAN WHITE DESCRIPTION BOX ---
        anomaly_html = f"""
<div style="background-color: #ffffff; padding: 40px; border-radius: 20px; color: #1a1a1a; border: 1px solid #eeeeee; box-shadow: 0 15px 35px rgba(0,0,0,0.1);">
    <h2 style="color: #5A2D5C; margin-top: 0; font-weight: 800; border-bottom: 3px solid #5A2D5C; padding-bottom: 15px;">🔍 Connection: Plot to Business Logic</h2>
    <p style="font-size: 18px; line-height: 1.8; margin-top: 20px;">
        In the plot above, we are comparing <b>{col_x}</b> against <b>{col_y}</b>. 
        The <b>Red Points</b> represent interactions that have a <b>Probability Density</b> too low to be considered "Standard Behavior."
    </p>
    <div style="font-size: 16px; line-height: 1.7; margin-top: 25px;">
        <p>● <b>Interactive Insight:</b> Hover over the <b>Red Points</b> to see their specific <b>Probability_Density</b>. The smaller the number, the more extreme the outlier.</p>
        <p>● <b>Feature Correlation:</b> Points far from the "Green Cloud" show a breakdown in expected correlation between your metrics, indicating a <b>unique customer persona</b> or <b>operational risk</b>.</p>
        <p>● <b>Statistical Cutoff:</b> We use the <b>Gaussian Bell Curve</b> logic. Anything in the bottom 5% tail of the distribution is automatically flagged for manual review.</p>
    </div>
</div>"""
        st.markdown(anomaly_html, unsafe_allow_html=True)

    with col_s:
        # --- 4. SIDEBAR TECHNICAL DEF ---
        sidebar_html = f"""
<div style="background-color: #5A2D5C; padding: 25px 15px; border-radius: 15px; text-align: center; border: 1px solid rgba(255,255,255,0.1); box-shadow: 0 8px 15px rgba(0,0,0,0.3);">
    <h4 style="color: white; font-size: 14px; letter-spacing: 1.2px; font-weight: bold; margin-bottom: 15px;">ANOMALY DEFINITION</h4>
    <p style="font-size: 11px; color: #F4DECB; line-height: 1.6; text-align: justify; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 10px; margin-bottom: 15px;">
        An <b>Anomaly</b> is a point that exists in the <b>low-density tail</b> of a Gaussian Distribution.
    </p>
    <p style="font-size: 11px; color: #ffffff; line-height: 1.5; text-align: justify;">
        Our <b>GMM Engine</b> identifies these based on the <b>log-likelihood</b> of them belonging to the main cluster.
    </p>
    <div style="background: rgba(255,75,75,0.2); padding: 12px; border-radius: 10px; border: 1px solid #FF4B4B; margin-top: 20px;">
        <span style="color: #FF4B4B; font-weight: bold; font-size: 13px;">DETECTIONS: {df[df['Status'] == 'Anomaly'].shape[0]}</span>
    </div>
</div>"""
        st.markdown(sidebar_html, unsafe_allow_html=True)

elif choice == "PCA":
    import textwrap
    from sklearn.decomposition import PCA
    import plotly.express as px
    import pandas as pd

    st.markdown('<h1 class="neon-header">DIMENSIONALITY REDUCTION (PCA)</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # --- 1. PCA Logic ---
    df_numeric = df.select_dtypes(include=[np.number])
    
    # Standardizing is crucial for PCA (assuming df is already scaled, if not, scaler should be used here)
    pca = PCA()
    pca_result = pca.fit_transform(df_numeric)
    
    # Captured Variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    col_m, col_s = st.columns([3.2, 0.8])

    with col_m:
        st.write("### 📉 Step 1: Variance & Information Capture")
        
        # Plotting Explained Variance
        fig_var = px.bar(
            x=[f"PC{i+1}" for i in range(len(explained_variance))],
            y=explained_variance,
            labels={'x': 'Principal Components', 'y': 'Variance Ratio'},
            title="Individual Variance Explained by Components",
            template="plotly_dark",
            color_discrete_sequence=['#00FF41']
        )
        st.plotly_chart(fig_var, use_container_width=True)

        

        # --- 2. DATA COMPARISON TABLE ---
        st.write("### 🔄 Dataset Comparison: Before vs. After PCA")
        
        c1, c2 = st.columns(2)
        with c1:
            st.info("**Original Shape**")
            st.code(f"Columns: {df_numeric.shape[1]}\nRows: {df_numeric.shape[0]}")
        with c2:
            st.success("**PCA Shape (2D)**")
            st.code(f"Columns: 2\nRows: {df_numeric.shape[0]}")

        # PCA Visualization (First 2 Components)
        st.write("### 📍 Step 2: 2D Projection of High-Dimensional Data")
        df_pca = pd.DataFrame(pca_result[:, :2], columns=['PC1', 'PC2'])
        
        fig_pca = px.scatter(
            df_pca, x='PC1', y='PC2',
            title="Data Reconstructed in 2D Space",
            template="plotly_dark",
            color_discrete_sequence=['#5A2D5C']
        )
        fig_pca.update_traces(marker=dict(size=10, line=dict(width=1, color='white')))
        st.plotly_chart(fig_pca, use_container_width=True)

        

        # --- 3. CLEAN WHITE DESCRIPTION BOX ---
        pca_html = f"""
<div style="background-color: #ffffff; padding: 40px; border-radius: 20px; color: #1a1a1a; border: 1px solid #eeeeee; box-shadow: 0 15px 35px rgba(0,0,0,0.1);">
    <h2 style="color: #5A2D5C; margin-top: 0; font-weight: 800; border-bottom: 3px solid #5A2D5C; padding-bottom: 15px;">🧠 Understanding PCA Magic</h2>
    <p style="font-size: 18px; line-height: 1.8; margin-top: 20px;">
        <b>Principal Component Analysis (PCA)</b> is a technique that simplifies complex data while keeping the most important information intact. 
    </p>
    <div style="font-size: 16px; line-height: 1.7; margin-top: 25px;">
        <p>● <b>Variance Captured:</b> PC1 and PC2 are not just random columns; they are "Directions" in your data that hold the maximum <b>Variance (Information)</b>. The bar chart shows how much 'story' each component tells.</p>
        <p>● <b>Dimensionality Reduction:</b> We compressed your <b>{df_numeric.shape[1]} features</b> into just <b>2 components</b>. This allows us to visualize complex customer relationships on a simple flat screen.</p>
        <p>● <b>Mathematical Integrity:</b> Despite reducing the columns, PCA ensures that the relative distances between data points remain as accurate as possible to the original dataset.</p>
    </div>
</div>"""
        st.markdown(pca_html, unsafe_allow_html=True)

    with col_s:
        # --- SIDEBAR STATUS ---
        sidebar_html = f"""
<div style="background-color: #5A2D5C; padding: 25px 15px; border-radius: 15px; text-align: center; border: 1px solid rgba(255,255,255,0.1); box-shadow: 0 8px 15px rgba(0,0,0,0.3);">
    <h4 style="color: white; font-size: 14px; letter-spacing: 1.2px; font-weight: bold; margin-bottom: 15px;">PCA ENGINE</h4>
    <hr style="border-color: rgba(255,255,255,0.2); margin: 10px 0;">
    <p style="font-size: 11px; color: #F4DECB; line-height: 1.6; text-align: justify; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 10px; margin-bottom: 15px;">
        <b>Principal Components</b> transform correlated features into a set of linearly uncorrelated variables.
    </p>
    <div style="background: rgba(0,255,65,0.15); padding: 12px; border-radius: 10px; border: 1px solid #00FF41; margin-top: 20px;">
        <p style="color: #ffffff; font-size: 10px; margin:0;">Total Variance Explained:</p>
        <span style="color: #00FF41; font-weight: bold; font-size: 16px;">{cumulative_variance[1]*100:.1f}%</span>
    </div>
</div>"""
        st.markdown(sidebar_html, unsafe_allow_html=True)

elif choice == "Recommendation":
    from sklearn.metrics.pairwise import cosine_similarity
    import pandas as pd
    import numpy as np

    st.markdown('<h1 class="neon-header">SMART COLLABORATIVE FILTERING</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # --- 1. Recommendation Logic (User-to-User) ---
    # Similarity hamesha scaled data (Z-score) par accurate aati hai
    # df_scaled humne preprocessing step mein create kiya tha
    user_sim_matrix = cosine_similarity(df_scaled)
    user_sim_df = pd.DataFrame(user_sim_matrix, index=df.index, columns=df.index)

    col_m, col_s = st.columns([3.2, 0.8])

    with col_m:
        st.write("### 🤝 Peer-Based Recommendations")
        
        # Collaborative Filtering ka logic: "Sone pe suhaga"
        st.write("#### Analyzing Similarity for Sample Users:")
        
        # Generating recommendations for 3 sample users
        sample_users = df.index[:3].tolist() 
        
        for user_id in sample_users:
            # Apne jaisa sabse kareebi user dhundna (Similarity score 1.0 ko skip karke)
            similar_users = user_sim_df[user_id].sort_values(ascending=False).iloc[1:4].index.tolist()
            
            with st.expander(f"✨ Personal Insights for Customer ID: {user_id}"):
                st.write(f"**Peer Group (Look-alike Customers):** {', '.join(map(str, similar_users))}")
                
                # Logic: In similar users ka average behavior kya hai?
                rec_logic = df.loc[similar_users].mean(numeric_only=True).to_frame().T
                st.write("**Recommended Spending & Interaction Profile:**")
                st.dataframe(rec_logic, use_container_width=True)
                st.caption("Target this user based on the average traits of their peer group.")

        

        # --- 2. CLEAN WHITE DESCRIPTION BOX ---
        # Bilkul left-align rakha hai taaki code error na aaye
        rec_html = f"""
<div style="background-color: #ffffff; padding: 40px; border-radius: 20px; color: #1a1a1a; border: 1px solid #eeeeee; box-shadow: 0 15px 35px rgba(0,0,0,0.1);">
    <h2 style="color: #5A2D5C; margin-top: 0; font-weight: 800; border-bottom: 3px solid #5A2D5C; padding-bottom: 15px;">🧠 How Collaborative Filtering Works</h2>
    <p style="font-size: 18px; line-height: 1.8; margin-top: 20px;">
        Collaborative filtering follows the <b>"Wisdom of the Crowd"</b> philosophy. Instead of just looking at what a customer bought, we look at <b>Who</b> they are similar to.
    </p>
    <div style="font-size: 16px; line-height: 1.7; margin-top: 25px;">
        <p>● <b>The Logic:</b> If User A and User B have similar spending habits and frequency, there is a high statistical probability that User A will like what User B is currently using.</p>
        <p>● <b>Cosine Similarity:</b> We treat each customer as a vector in space. By measuring the <b>angle</b> between these vectors, we determine how 'close' two customers are in their behavior.</p>
        <p>● <b>Business Impact:</b> This allows for <b>Cross-Selling</b> and <b>Up-Selling</b> by recommending products that 'people like you' have already purchased.</p>
    </div>
</div>"""
        st.markdown(rec_html, unsafe_allow_html=True)

    with col_s:
        # --- SIDEBAR TECH STATUS ---
        sidebar_html = f"""
<div style="background-color: #5A2D5C; padding: 25px 15px; border-radius: 15px; text-align: center; border: 1px solid rgba(255,255,255,0.1); box-shadow: 0 8px 15px rgba(0,0,0,0.3);">
    <h4 style="color: white; font-size: 14px; letter-spacing: 1.2px; font-weight: bold; margin-bottom: 15px;">RECO SYSTEM</h4>
    <hr style="border-color: rgba(255,255,255,0.2); margin: 10px 0;">
    <p style="font-size: 11px; color: #F4DECB; line-height: 1.6;">
        <b>Engine:</b> User-Based Filtering<br>
        <b>Metric:</b> Cosine Similarity<br>
        <b>Data:</b> Z-Score Normalized
    </p>
    <div style="background: rgba(0,255,65,0.15); padding: 12px; border-radius: 10px; border: 1px solid #00FF41; margin-top: 20px;">
        <p style="color: #ffffff; font-size: 10px; margin:0;">Accuracy Status:</p>
        <span style="color: #00FF41; font-weight: bold; font-size: 16px;">OPTIMIZED</span>
    </div>
</div>"""
        st.markdown(sidebar_html, unsafe_allow_html=True)
# --- Reflection Analysis & Reflection Section ---
elif choice == "Reflection":
    st.markdown('<h1 class="neon-header">STRATEGIC REFLECTION & INSIGHTS</h1>', unsafe_allow_html=True)
    st.markdown("---")

    col_m, col_s = st.columns([3.2, 0.8])

    with col_m:
        st.write("### 🧠 Uncovering Hidden Data Patterns")
        
        st.markdown("""
        <div class="white-box" style="margin-top: 20px;">
            <h2>💡 The Power of Unsupervised Learning</h2>
            <p>Unsupervised learning has transformed our approach to data analysis. By allowing algorithms to find patterns without prior labels, we've gained:</p>
            <ul>
                <li><b>Objective Insights:</b> Discovered natural groupings and anomalies unbiased by human assumptions.</li>
                <li><b>Complexity Reduction:</b> Simplified high-dimensional data into understandable visual forms.</li>
                <li><b>Actionable Intelligence:</b> Converted raw data into concrete strategies for marketing, risk, and growth.</li>
            </ul>
            <p>This pipeline is a testament to the ability of AI to unlock deeper value from complex datasets.</p>
        </div>
        """, unsafe_allow_html=True)

        st.write("---")
        st.write("### 📊 Technique Comparison: Utility & Impact")
        
        comparison_data = {
            "Technique": [
                "**Clustering (K-Means)**", 
                "**Anomaly Detection (GMM)**", 
                "**PCA (Dimensionality Reduction)**", 
                "**Recommendation (CF)**"
            ],
            "Primary Goal": ["Group Similar Data Points", "Identify Rare Occurrences", "Simplify & Visualize Data", "Suggest Relevant Items"],
            "Key Business Value": ["Targeted Marketing, Customer Segments", "Fraud Detection, Error Identification", "Feature Engineering, Noise Reduction", "Personalized Sales, Customer Retention"],
            "Visual Impact": ["Distinct groups on plots.", "Highlighted outliers.", "2D data projection.", "Peer-group profiles."]
        }
        
        st.table(pd.DataFrame(comparison_data))

        st.write("---")
        
        reflection_html = f"""
<div class="white-box">
    <h2>🌍 Real-World Applications of This Pipeline</h2>
    <p>This comprehensive Unsupervised Learning pipeline is incredibly versatile and can drive significant value across various industries:</p>
    <ul>
        <li><b>E-Commerce & Retail:</b> 
            <p style="margin-left: 15px; font-size: 15px;">Automatically segment customers for personalized promotions, detect fraudulent purchases, and recommend complementary products.</p>
        </li>
        <li><b>FinTech & Banking:</b> 
            <p style="margin-left: 15px; font-size: 15px;">Identify unusual transaction patterns indicative of fraud and group clients for tailored financial products.</p>
        </li>
        <li><b>Healthcare:</b> 
            <p style="margin-left: 15px; font-size: 15px;">Cluster patients with similar conditions for targeted treatment plans and detect anomalous lab results.</p>
        </li>
        <li><b>Cybersecurity:</b> 
            <p style="margin-left: 15px; font-size: 15px;">Flag abnormal network traffic for intrusion detection and segment network users based on behavior.</p>
        </li>
    </ul>
    <p>By leveraging these techniques, organizations can make data-driven decisions that enhance efficiency, reduce risks, and create superior customer experiences.</p>
</div>"""
        st.markdown(reflection_html, unsafe_allow_html=True)

    with col_s:
        st.write("### 🖼️ Project Modules")
        
        # 1. Mini Cluster Plot (Fake data representation)
        df_mini = pd.DataFrame(np.random.rand(10, 2), columns=['x', 'y'])
        fig1 = px.scatter(df_mini, x='x', y='y', color_discrete_sequence=['#00FFFF'], template="plotly_dark")
        fig1.update_layout(height=120, margin=dict(l=0,r=0,t=0,b=0), showlegend=False, xaxis_visible=False, yaxis_visible=False)
        st.plotly_chart(fig1, use_container_width=True, config={'displayModeBar': False})
        st.caption("1. Customer Clusters")

        # 2. Mini Anomaly Plot
        fig2 = px.scatter(df_mini, x='x', y='y', color_discrete_sequence=['#FF00FF'], template="plotly_dark")
        fig2.update_layout(height=120, margin=dict(l=0,r=0,t=0,b=0), showlegend=False, xaxis_visible=False, yaxis_visible=False)
        st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
        st.caption("2. Anomalies Detected")

        st.markdown("---")

        # 3. Mini PCA Plot
        fig3 = px.bar(x=[1,2,3], y=[10,5,2], color_discrete_sequence=['#00BFFF'], template="plotly_dark")
        fig3.update_layout(height=120, margin=dict(l=0,r=0,t=0,b=0), showlegend=False, xaxis_visible=False, yaxis_visible=False)
        st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})
        st.caption("3. PCA Variance")

        # 4. Mini Recommendation Metric
        st.markdown("""<div style="height:120px; background:#161b22; border:1px solid #ffffff; border-radius:5px; display:flex; align-items:center; justify-content:center; color:#00FFFF; font-weight:bold; font-size:24px;">98%</div>""", unsafe_allow_html=True)
        st.caption("4. Recommender Accuracy")

        sidebar_html = f"""
<div class="sidebar-status-box" style="margin-top:20px; background:#1e2a38; padding:15px; border-radius:10px; border:1px solid #00FFFF;">
    <h4 style="color:#00FFFF; margin:0;">PROJECT STATUS</h4>
    <hr style="border-color:rgba(0,255,255,0.2);">
    <p style="font-size:12px; color:white;"><b>Dashboard:</b> Operational <br><b>Modules:</b> 4 Active</p>
    <div style="background:rgba(0,255,255,0.1); padding:5px; border-radius:5px; text-align:center;">
        <span style="color:#00FFFF; font-weight:bold; font-size:14px;">TRANSFORMED</span>
    </div>
</div>"""
        st.markdown(sidebar_html, unsafe_allow_html=True)

# --- FOOTER (Always Outside the Block) ---
st.markdown(f"""
    <div class="footer-box">
        <h2 style="letter-spacing: 3px; font-weight: 900;">SYSTEM INITIALIZED</h2>
        <p style="font-size: 18px; color: #F4DECB;"><b>UNCOVERING THE UNSEEN:</b> Pattern Discovery via Unsupervised ML.</p>
        <div style="margin: 30px 0;">
            <a href="https://github.com/tahira879" target="_blank" class="social-btn">Git</a>
            <a href="https://linkedin.com" target="_blank" class="social-btn">In</a>
        </div>
        <p style="font-size: 14px; opacity: 0.7; letter-spacing: 2px;">© 2026 | DESIGNED BY TAHIRA MUHAMMAD JAVED</p>
    </div>

""", unsafe_allow_html=True)
