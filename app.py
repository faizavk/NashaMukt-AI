import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from model_logic import get_trained_model

st.set_page_config(page_title="NashaMukt AI", layout="wide")
st.title("🇮🇳 NashaMukt AI: National Policy Optimizer")
st.write("Leveraging AI for Atmanirbhar Bharat | India AI Impact Summit 2026")

# Load Engine
model, importances, r2, df_clean = get_trained_model()

if model is None:
    st.error("Could not find 'GYTS4.xls'. Please ensure it is in the same folder as this script.")
else:
    # Sidebar Navigation
    page = st.sidebar.selectbox("Choose Analysis Level", ["National Overview", "State-Specific Prediction"])

    if page == "National Overview":
        st.header("📊 Overall Country Visualizations")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("State-wise Tobacco Prevalence")
            # Show top 20 states for clarity
            df_total = df_clean[df_clean['Area'] == 'Total'].sort_values('Current tobacco users (%)', ascending=False)
            fig, ax = plt.subplots(figsize=(8, 10))
            sns.barplot(x='Current tobacco users (%)', y='State/UT', data=df_total, palette='flare', ax=ax)
            st.pyplot(fig)

        with col2:
            st.subheader("Urban vs. Rural Distribution")
            df_ur = df_clean[df_clean['Area'].isin(['Urban', 'Rural'])]
            fig2, ax2 = plt.subplots()
            sns.boxplot(x='Area', y='Current tobacco users (%)', data=df_ur, palette='pastel', ax=ax2)
            st.pyplot(fig2)
            st.info("The wide range in Rural areas suggests localized drivers of addiction that require targeted AI intervention.")

    else:
        st.header("🎯 State-Level Predictive Analysis")
        selected_state = st.selectbox("Select a State/UT", df_clean['State/UT'].unique())
        
        # Filter data for selected state
        state_data = df_clean[df_clean['State/UT'] == selected_state].iloc[0]
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Youth Usage Rate", f"{state_data['Current tobacco users (%)']}%")
            st.subheader("💡 AI Insight")
            top_driver = importances.index[0]
            st.write(f"In {selected_state}, the model identifies **'{top_driver}'** as the most significant variable.")
            st.success("Recommendation: Prioritize enforcement of COTPA Section 4 in public spaces.")

        with c2:
            st.subheader("Model Feature Importance (Technical)")
            fig3, ax3 = plt.subplots()
            importances.head(10).plot(kind='barh', ax=ax3, color='teal')
            ax3.invert_yaxis()
            st.pyplot(fig3)

    st.divider()
    st.caption(f"Model Accuracy (R²): {r2:.4f} | Dataset: Global Youth Tobacco Survey (GYTS-4)")