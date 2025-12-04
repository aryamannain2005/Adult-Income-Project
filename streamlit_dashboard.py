# Interactive Dashboard for Adult Income Analysis
# Run with: streamlit run streamlit_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Adult Income Analyzer", layout="wide")

# Title and description
st.title("🏦 Adult Income Dataset Analysis Dashboard")
st.markdown("""
Solving 5 Real-World Problems:
1. **Financial Inclusion** - Fair credit assessment
2. **HR Salary Benchmarking** - Pay equity analysis
3. **Social Policy** - Welfare program optimization
4. **Career Guidance** - Income prediction for career planning
5. **Economic Research** - Income inequality analysis
""")

@st.cache_data
def load_data():
    try:
        from ucimlrepo import fetch_ucirepo
        adult = fetch_ucirepo(id=2)
        data = pd.concat([adult.data.features, adult.data.targets], axis=1)
        return data
    except:
        st.error("Please install ucimlrepo: pip install ucimlrepo")
        return None

def main():
    # Load data
    data = load_data()
    if data is None:
        return
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose Analysis", [
        "📊 Data Overview",
        "🏦 Financial Inclusion",
        "💼 HR Salary Analysis",
        "🏛️ Social Policy",
        "🎯 Career Guidance",
        "📈 Economic Research",
        "🤖 Income Predictor"
    ])
    
    if page == "📊 Data Overview":
        st.header("Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(data):,}")
        with col2:
            st.metric("Features", len(data.columns) - 1)
        with col3:
            high_income_rate = (data['income'] == '>50K').mean() * 100
            st.metric("High Income Rate", f"{high_income_rate:.1f}%")
        with col4:
            missing_rate = data.isnull().sum().sum() / (len(data) * len(data.columns)) * 100
            st.metric("Missing Data Rate", f"{missing_rate:.1f}%")
        
        # Income distribution
        fig = px.pie(data, names='income', title='Income Distribution')
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature correlation heatmap
        st.subheader("Numerical Features Correlation")
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        corr_matrix = data[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    
    elif page == "🏦 Financial Inclusion":
        st.header("Financial Inclusion Analysis")
        st.markdown("**Objective**: Ensure fair credit assessment across demographics")
        
        # Gender analysis
        gender_income = pd.crosstab(data['sex'], data['income'], normalize='index') * 100
        fig = px.bar(gender_income, title='Income Distribution by Gender (%)')
        st.plotly_chart(fig, use_container_width=True)
        
        # Race analysis
        race_income = pd.crosstab(data['race'], data['income'], normalize='index') * 100
        fig = px.bar(race_income, title='Income Distribution by Race (%)')
        st.plotly_chart(fig, use_container_width=True)
        
        # Key insights
        st.subheader("Key Insights for Financial Inclusion")
        st.write("• Significant income disparities exist across demographic groups")
        st.write("• Models should be regularly audited for bias")
        st.write("• Alternative data sources may help underrepresented groups")
    
    elif page == "💼 HR Salary Analysis":
        st.header("HR Salary Benchmarking")
        st.markdown("**Objective**: Identify pay gaps and ensure equitable compensation")
        
        # Education vs Income
        edu_income = pd.crosstab(data['education'], data['income'], normalize='index') * 100
        fig = px.bar(edu_income, title='High Income Rate by Education Level (%)')
        st.plotly_chart(fig, use_container_width=True)
        
        # Work hours analysis
        fig = px.box(data, x='income', y='hours-per-week', title='Work Hours Distribution by Income')
        st.plotly_chart(fig, use_container_width=True)
        
        # Occupation analysis
        occ_income = pd.crosstab(data['occupation'], data['income'], normalize='index') * 100
        top_occupations = data['occupation'].value_counts().head(10).index
        occ_subset = occ_income.loc[top_occupations]
        fig = px.bar(occ_subset, title='High Income Rate by Top Occupations (%)')
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "🏛️ Social Policy":
        st.header("Social Policy Optimization")
        st.markdown("**Objective**: Target welfare programs effectively")
        
        # Age distribution by income
        fig = px.histogram(data, x='age', color='income', title='Age Distribution by Income', 
                          marginal='box', nbins=30)
        st.plotly_chart(fig, use_container_width=True)
        
        # Marital status analysis
        marital_income = pd.crosstab(data['marital-status'], data['income'], normalize='index') * 100
        fig = px.bar(marital_income, title='High Income Rate by Marital Status (%)')
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk factors identification
        st.subheader("High-Risk Groups for Low Income")
        low_income_rate = pd.crosstab(data['education'], data['income'], normalize='index')['<=50K'] * 100
        high_risk_education = low_income_rate.sort_values(ascending=False).head(5)
        st.write("Education levels with highest low-income risk:")
        for edu, rate in high_risk_education.items():
            st.write(f"• {edu}: {rate:.1f}% low income rate")
    
    elif page == "🎯 Career Guidance":
        st.header("Career Guidance Insights")
        st.markdown("**Objective**: Provide data-driven career advice")
        
        # Feature importance (simplified)
        st.subheader("Factors Most Important for High Income")
        
        # Education impact
        edu_impact = pd.crosstab(data['education'], data['income'], normalize='index')['>50K'] * 100
        edu_impact = edu_impact.sort_values(ascending=False)
        
        fig = px.bar(x=edu_impact.values, y=edu_impact.index, orientation='h',
                    title='High Income Probability by Education Level (%)')
        st.plotly_chart(fig, use_container_width=True)
        
        # Career recommendations
        st.subheader("Career Recommendations")
        st.write("**High-Income Probability Occupations:**")
        occ_income_rate = pd.crosstab(data['occupation'], data['income'], normalize='index')['>50K'] * 100
        top_occupations = occ_income_rate.sort_values(ascending=False).head(5)
        for occ, rate in top_occupations.items():
            st.write(f"• {occ}: {rate:.1f}% high income rate")
    
    elif page == "📈 Economic Research":
        st.header("Economic Research Insights")
        st.markdown("**Objective**: Understand income inequality patterns")
        
        # Income inequality metrics
        st.subheader("Income Inequality Analysis")
        
        # Gender pay gap
        gender_gap = pd.crosstab(data['sex'], data['income'], normalize='index')['>50K'] * 100
        male_rate = gender_gap.get('Male', 0)
        female_rate = gender_gap.get('Female', 0)
        gap = male_rate - female_rate
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Male High Income Rate", f"{male_rate:.1f}%")
        with col2:
            st.metric("Female High Income Rate", f"{female_rate:.1f}%")
        with col3:
            st.metric("Gender Gap", f"{gap:.1f}pp")
        
        # Education ROI analysis
        st.subheader("Education Return on Investment")
        edu_levels = ['HS-grad', 'Some-college', 'Bachelors', 'Masters', 'Doctorate']
        edu_data = data[data['education'].isin(edu_levels)]
        edu_income_rate = pd.crosstab(edu_data['education'], edu_data['income'], normalize='index')['>50K'] * 100
        
        fig = px.line(x=edu_levels, y=[edu_income_rate.get(edu, 0) for edu in edu_levels],
                     title='High Income Rate by Education Level')
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "🤖 Income Predictor":
        st.header("Income Prediction Tool")
        st.markdown("**Objective**: Predict income based on individual characteristics")
        
        # Input form
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", 17, 90, 35)
            education = st.selectbox("Education", data['education'].unique())
            hours_per_week = st.slider("Hours per Week", 1, 99, 40)
            sex = st.selectbox("Gender", data['sex'].unique())
        
        with col2:
            workclass = st.selectbox("Work Class", data['workclass'].unique())
            occupation = st.selectbox("Occupation", data['occupation'].unique())
            marital_status = st.selectbox("Marital Status", data['marital-status'].unique())
            race = st.selectbox("Race", data['race'].unique())
        
        if st.button("Predict Income"):
            # Simplified prediction logic
            # In practice, you'd use a trained model with proper preprocessing
            
            # Simple rule-based prediction for demo
            score = 0
            
            # Education scoring
            edu_scores = {'Doctorate': 40, 'Masters': 35, 'Bachelors': 30, 'Some-college': 20, 'HS-grad': 15}
            score += edu_scores.get(education, 10)
            
            # Age scoring
            if 30 <= age <= 55:
                score += 20
            elif age > 55:
                score += 15
            
            # Hours scoring
            if hours_per_week >= 40:
                score += 15
            
            # Gender scoring (reflecting real-world bias in data)
            if sex == 'Male':
                score += 10
            
            # Prediction
            prediction = ">50K" if score >= 50 else "<=50K"
            confidence = min(score * 2, 100)
            
            st.success(f"Predicted Income: {prediction}")
            st.info(f"Confidence: {confidence:.0f}%")
            
            # Explanation
            st.subheader("Prediction Factors")
            st.write(f"• Education contributes significantly to income potential")
            st.write(f"• Age and work experience matter")
            st.write(f"• Work hours indicate commitment level")
            st.write(f"• Note: This model reflects historical biases in the data")

if __name__ == "__main__":
    main()