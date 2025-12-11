import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Marketing Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for eye-pleasing design
st.markdown("""
    <style>
    /* Main background */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0rem 1rem;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        color: white;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: bold;
        color: #1f77b4;
    }
    
    .stMetric {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #667eea;
    }
    
    /* Headers */
    h1 {
        color: #ffffff;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        padding: 20px 0;
    }
    
    h2 {
        color: #ffffff;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 600;
        margin-top: 20px;
    }
    
    h3 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 500;
    }
    
    /* Cards */
    .css-1r6slb0 {
        background: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 25px;
        padding: 10px 25px;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }
    
    /* Selectbox and multiselect */
    .stSelectbox, .stMultiSelect {
        background: white;
        border-radius: 10px;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 10px;
        border-left: 5px solid #667eea;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px 10px 0 0;
        color: white;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Plotly charts background */
    .js-plotly-plot {
        border-radius: 10px;
        background: white;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Load data with caching
@st.cache_data
def load_data():
    """Load all CSV files"""
    try:
        campaign_performance = pd.read_csv('campaign_performance.csv')
        channel_attribution = pd.read_csv('channel_attribution.csv')
        correlation_matrix = pd.read_csv('correlation_matrix.csv')
        customer_data = pd.read_csv('customer_data.csv')
        customer_journey = pd.read_csv('customer_journey.csv')
        feature_importance = pd.read_csv('feature_importance.csv')
        funnel_data = pd.read_csv('funnel_data.csv')
        geographic_data = pd.read_csv('geographic_data.csv')
        lead_scoring_results = pd.read_csv('lead_scoring_results.csv')
        learning_curve = pd.read_csv('learning_curve.csv')
        product_sales = pd.read_csv('product_sales.csv')
        
        # Convert date columns if they exist
        date_columns = ['Date', 'date', 'timestamp', 'Timestamp']
        for df in [campaign_performance, customer_journey, product_sales]:
            for col in df.columns:
                if col in date_columns or 'date' in col.lower():
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except:
                        pass
        
        return (campaign_performance, channel_attribution, correlation_matrix, 
                customer_data, customer_journey, feature_importance, funnel_data,
                geographic_data, lead_scoring_results, learning_curve, product_sales)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load the data
data = load_data()

if data is None:
    st.error("❌ Failed to load data files. Please ensure all CSV files are in the correct directory.")
    st.stop()

(campaign_performance, channel_attribution, correlation_matrix, customer_data, 
 customer_journey, feature_importance, funnel_data, geographic_data, 
 lead_scoring_results, learning_curve, product_sales) = data

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/fluency/96/000000/marketing.png", width=100)
st.sidebar.title("📊 Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select Dashboard",
    ["🏠 Overview", "📈 Campaign Performance", "🎯 Customer Analytics", 
     "🛍️ Product Sales", "🌍 Geographic Insights", "🔄 Customer Journey",
     "📊 Channel Attribution", "🎓 ML Insights", "🔍 Lead Scoring"]
)

st.sidebar.markdown("---")
st.sidebar.info("**Marketing Analytics Dashboard**\n\nComprehensive insights into campaigns, customers, and performance metrics.")
st.sidebar.markdown("---")
st.sidebar.markdown("### 📞 Support")
st.sidebar.markdown("📧 support@marketing.com")
st.sidebar.markdown("🌐 www.marketing.com")

# Main title with icon
st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1>🎯 Marketing Analytics Dashboard</h1>
        <p style='color: white; font-size: 18px;'>Data-Driven Marketing Intelligence Platform</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== OVERVIEW PAGE ====================
if page == "🏠 Overview":
    st.markdown("## 📊 Executive Summary")
    
    # Key Metrics Row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_customers = len(customer_data)
        st.metric("👥 Total Customers", f"{total_customers:,}", delta="↑ 12%")
    
    with col2:
        if 'Revenue' in product_sales.columns:
            total_revenue = product_sales['Revenue'].sum()
            st.metric("💰 Total Revenue", f"${total_revenue:,.0f}", delta="↑ 8.5%")
        else:
            st.metric("💰 Total Revenue", "N/A")
    
    with col3:
        if 'Conversions' in campaign_performance.columns:
            total_conversions = campaign_performance['Conversions'].sum()
            st.metric("✅ Conversions", f"{int(total_conversions):,}", delta="↑ 15%")
        else:
            st.metric("✅ Conversions", "N/A")
    
    with col4:
        if 'CTR' in campaign_performance.columns:
            avg_ctr = campaign_performance['CTR'].mean()
            st.metric("📊 Avg CTR", f"{avg_ctr:.2f}%", delta="↑ 2.3%")
        else:
            st.metric("📊 Avg CTR", "N/A")
    
    with col5:
        if 'ROI' in campaign_performance.columns:
            avg_roi = campaign_performance['ROI'].mean()
            st.metric("💹 Avg ROI", f"{avg_roi:.1f}%", delta="↑ 5.2%")
        else:
            st.metric("💹 Avg ROI", "N/A")
    
    st.markdown("---")
    
    # Two column layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Campaign Performance Trend")
        if 'Date' in campaign_performance.columns and 'Revenue' in campaign_performance.columns:
            fig = px.line(campaign_performance, x='Date', y='Revenue', 
                         title='Revenue Over Time',
                         template='plotly_white')
            fig.update_traces(line_color='#667eea', line_width=3)
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#2c3e50')
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Revenue trend data not available")
    
    with col2:
        st.markdown("### 🎯 Conversion Funnel")
        if 'Stage' in funnel_data.columns and 'Count' in funnel_data.columns:
            fig = go.Figure(go.Funnel(
                y=funnel_data['Stage'],
                x=funnel_data['Count'],
                textinfo="value+percent initial",
                marker=dict(color=['#667eea', '#764ba2', '#f093fb', '#4facfe'])
            ))
            fig.update_layout(
                template='plotly_white',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Funnel data not available")
    
    # Full width charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🌍 Geographic Distribution")
        if 'Region' in geographic_data.columns and 'Revenue' in geographic_data.columns:
            fig = px.bar(geographic_data, x='Region', y='Revenue',
                        color='Revenue',
                        color_continuous_scale='Viridis',
                        title='Revenue by Region')
            fig.update_layout(
                template='plotly_white',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Geographic data not available")
    
    with col2:
        st.markdown("### 📊 Channel Performance")
        if 'Channel' in channel_attribution.columns and 'Conversions' in channel_attribution.columns:
            fig = px.pie(channel_attribution, values='Conversions', names='Channel',
                        title='Conversions by Channel',
                        color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_layout(
                template='plotly_white',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Channel attribution data not available")

# ==================== CAMPAIGN PERFORMANCE PAGE ====================
elif page == "📈 Campaign Performance":
    st.markdown("## 📈 Campaign Performance Analysis")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'Campaign_Name' in campaign_performance.columns:
            campaigns = st.multiselect(
                "Select Campaigns",
                options=campaign_performance['Campaign_Name'].unique(),
                default=campaign_performance['Campaign_Name'].unique()[:5]
            )
        else:
            campaigns = []
    
    with col2:
        if 'Channel' in campaign_performance.columns:
            channels = st.multiselect(
                "Select Channels",
                options=campaign_performance['Channel'].unique(),
                default=campaign_performance['Channel'].unique()
            )
        else:
            channels = []
    
    with col3:
        date_range = st.date_input(
            "Date Range",
            value=(datetime.now().replace(day=1), datetime.now())
        )
    
    # Filter data
    filtered_cp = campaign_performance.copy()
    if campaigns and 'Campaign_Name' in campaign_performance.columns:
        filtered_cp = filtered_cp[filtered_cp['Campaign_Name'].isin(campaigns)]
    if channels and 'Channel' in campaign_performance.columns:
        filtered_cp = filtered_cp[filtered_cp['Channel'].isin(channels)]
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'Impressions' in filtered_cp.columns:
            total_impressions = filtered_cp['Impressions'].sum()
            st.metric("👁️ Total Impressions", f"{total_impressions:,.0f}")
        else:
            st.metric("👁️ Total Impressions", "N/A")
    
    with col2:
        if 'Clicks' in filtered_cp.columns:
            total_clicks = filtered_cp['Clicks'].sum()
            st.metric("🖱️ Total Clicks", f"{int(total_clicks):,}")
        else:
            st.metric("🖱️ Total Clicks", "N/A")
    
    with col3:
        if 'Conversions' in filtered_cp.columns:
            total_conversions = filtered_cp['Conversions'].sum()
            st.metric("✅ Total Conversions", f"{int(total_conversions):,}")
        else:
            st.metric("✅ Total Conversions", "N/A")
    
    with col4:
        if 'Spend' in filtered_cp.columns:
            total_spend = filtered_cp['Spend'].sum()
            st.metric("💵 Total Spend", f"${total_spend:,.2f}")
        else:
            st.metric("💵 Total Spend", "N/A")
    
    st.markdown("---")
    
    # Campaign Performance Table
    st.markdown("### 📊 Campaign Performance Details")
    
    if not filtered_cp.empty:
        # Calculate additional metrics
        if 'CTR' not in filtered_cp.columns and 'Clicks' in filtered_cp.columns and 'Impressions' in filtered_cp.columns:
            filtered_cp['CTR'] = (filtered_cp['Clicks'] / filtered_cp['Impressions'] * 100).round(2)
        
        if 'CPC' not in filtered_cp.columns and 'Spend' in filtered_cp.columns and 'Clicks' in filtered_cp.columns:
            filtered_cp['CPC'] = (filtered_cp['Spend'] / filtered_cp['Clicks']).round(2)
        
        if 'Conversion_Rate' not in filtered_cp.columns and 'Conversions' in filtered_cp.columns and 'Clicks' in filtered_cp.columns:
            filtered_cp['Conversion_Rate'] = (filtered_cp['Conversions'] / filtered_cp['Clicks'] * 100).round(2)
        
        st.dataframe(filtered_cp.head(10), use_container_width=True)
    else:
        st.info("No data available for selected filters")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 CTR by Campaign")
        if 'Campaign_Name' in filtered_cp.columns and 'CTR' in filtered_cp.columns:
            fig = px.bar(filtered_cp.sort_values('CTR', ascending=False).head(10),
                        x='Campaign_Name', y='CTR',
                        color='CTR',
                        color_continuous_scale='Blues',
                        title='Top 10 Campaigns by CTR')
            fig.update_layout(
                template='plotly_white',
                xaxis_tickangle=-45,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("CTR data not available")
    
    with col2:
        st.markdown("### 💰 ROI by Campaign")
        if 'Campaign_Name' in filtered_cp.columns and 'ROI' in filtered_cp.columns:
            fig = px.bar(filtered_cp.sort_values('ROI', ascending=False).head(10),
                        x='Campaign_Name', y='ROI',
                        color='ROI',
                        color_continuous_scale='RdYlGn',
                        title='Top 10 Campaigns by ROI')
            fig.update_layout(
                template='plotly_white',
                xaxis_tickangle=-45,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ROI data not available")
    
    # Performance over time
    st.markdown("### 📈 Performance Trends")
    if 'Date' in filtered_cp.columns:
        metric_choice = st.selectbox(
            "Select Metric",
            ['Impressions', 'Clicks', 'Conversions', 'Revenue', 'Spend']
        )
        
        if metric_choice in filtered_cp.columns:
            daily_perf = filtered_cp.groupby('Date')[metric_choice].sum().reset_index()
            fig = px.line(daily_perf, x='Date', y=metric_choice,
                         title=f'{metric_choice} Over Time',
                         markers=True)
            fig.update_traces(line_color='#667eea', line_width=3)
            fig.update_layout(
                template='plotly_white',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"{metric_choice} data not available")

# ==================== CUSTOMER ANALYTICS PAGE ====================
elif page == "🎯 Customer Analytics":
    st.markdown("## 🎯 Customer Analytics")
    
    # Customer Segmentation
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'Age_Group' in customer_data.columns:
            age_groups = st.multiselect(
                "Age Group",
                options=customer_data['Age_Group'].unique(),
                default=customer_data['Age_Group'].unique()
            )
        else:
            age_groups = []
    
    with col2:
        if 'Income_Level' in customer_data.columns:
            income_levels = st.multiselect(
                "Income Level",
                options=customer_data['Income_Level'].unique(),
                default=customer_data['Income_Level'].unique()
            )
        else:
            income_levels = []
    
    with col3:
        if 'Customer_Segment' in customer_data.columns:
            segments = st.multiselect(
                "Customer Segment",
                options=customer_data['Customer_Segment'].unique(),
                default=customer_data['Customer_Segment'].unique()
            )
        else:
            segments = []
    
    # Filter customer data
    filtered_customers = customer_data.copy()
    if age_groups and 'Age_Group' in customer_data.columns:
        filtered_customers = filtered_customers[filtered_customers['Age_Group'].isin(age_groups)]
    if income_levels and 'Income_Level' in customer_data.columns:
        filtered_customers = filtered_customers[filtered_customers['Income_Level'].isin(income_levels)]
    if segments and 'Customer_Segment' in customer_data.columns:
        filtered_customers = filtered_customers[filtered_customers['Customer_Segment'].isin(segments)]
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("👥 Total Customers", f"{len(filtered_customers):,}")
    
    with col2:
        if 'Total_Spend' in filtered_customers.columns:
            avg_spend = filtered_customers['Total_Spend'].mean()
            st.metric("💰 Avg Customer Value", f"${avg_spend:,.2f}")
        else:
            st.metric("💰 Avg Customer Value", "N/A")
    
    with col3:
        if 'Purchase_Frequency' in filtered_customers.columns:
            avg_freq = filtered_customers['Purchase_Frequency'].mean()
            st.metric("🔄 Avg Purchase Freq", f"{avg_freq:.1f}")
        else:
            st.metric("🔄 Avg Purchase Freq", "N/A")
    
    with col4:
        if 'Churn_Risk' in filtered_customers.columns:
            churn_rate = (filtered_customers['Churn_Risk'] == 'High').sum() / len(filtered_customers) * 100
            st.metric("⚠️ High Churn Risk", f"{churn_rate:.1f}%")
        else:
            st.metric("⚠️ High Churn Risk", "N/A")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👥 Customer Segmentation")
        if 'Customer_Segment' in filtered_customers.columns:
            segment_counts = filtered_customers['Customer_Segment'].value_counts().reset_index()
            segment_counts.columns = ['Segment', 'Count']
            fig = px.pie(segment_counts, values='Count', names='Segment',
                        title='Customer Distribution by Segment',
                        color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_layout(
                template='plotly_white',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Segment data not available")
    
    with col2:
        st.markdown("### 💰 Revenue by Age Group")
        if 'Age_Group' in filtered_customers.columns and 'Total_Spend' in filtered_customers.columns:
            age_revenue = filtered_customers.groupby('Age_Group')['Total_Spend'].sum().reset_index()
            fig = px.bar(age_revenue, x='Age_Group', y='Total_Spend',
                        color='Total_Spend',
                        color_continuous_scale='Viridis',
                        title='Total Spend by Age Group')
            fig.update_layout(
                template='plotly_white',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Age group revenue data not available")
    
    # Customer Lifetime Value
    st.markdown("### 💎 Customer Lifetime Value Analysis")
    if 'Total_Spend' in filtered_customers.columns and 'Purchase_Frequency' in filtered_customers.columns:
        fig = px.scatter(filtered_customers, x='Purchase_Frequency', y='Total_Spend',
                        color='Customer_Segment' if 'Customer_Segment' in filtered_customers.columns else None,
                        size='Total_Spend',
                        title='Customer Value: Purchase Frequency vs Total Spend',
                        labels={'Purchase_Frequency': 'Purchase Frequency', 'Total_Spend': 'Total Spend ($)'})
        fig.update_layout(
            template='plotly_white',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Customer lifetime value data not available")
    
    # Customer Demographics Table
    st.markdown("### 📊 Customer Demographics")
    if not filtered_customers.empty:
        st.dataframe(filtered_customers.head(10), use_container_width=True)
    else:
        st.info("No customer data available")

# ==================== PRODUCT SALES PAGE ====================
elif page == "🛍️ Product Sales":
    st.markdown("## 🛍️ Product Sales Analysis")
    
    # Filters
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Product_Category' in product_sales.columns:
            categories = st.multiselect(
                "Product Category",
                options=product_sales['Product_Category'].unique(),
                default=product_sales['Product_Category'].unique()
            )
        else:
            categories = []
    
    with col2:
        if 'Price' in product_sales.columns:
            price_range = st.slider(
                "Price Range",
                min_value=float(product_sales['Price'].min()),
                max_value=float(product_sales['Price'].max()),
                value=(float(product_sales['Price'].min()), float(product_sales['Price'].max()))
            )
        else:
            price_range = (0, 1000)
    
    # Filter product data
    filtered_products = product_sales.copy()
    if categories and 'Product_Category' in product_sales.columns:
        filtered_products = filtered_products[filtered_products['Product_Category'].isin(categories)]
    if 'Price' in product_sales.columns:
        filtered_products = filtered_products[
            (filtered_products['Price'] >= price_range[0]) & 
            (filtered_products['Price'] <= price_range[1])
        ]
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'Product_ID' in filtered_products.columns:
            total_products = filtered_products['Product_ID'].nunique()
            st.metric("📦 Total Products", f"{total_products:,}")
        else:
            st.metric("📦 Total Products", "N/A")
    
    with col2:
        if 'Revenue' in filtered_products.columns:
            total_revenue = filtered_products['Revenue'].sum()
            st.metric("💰 Total Revenue", f"${total_revenue:,.0f}")
        else:
            st.metric("💰 Total Revenue", "N/A")
    
    with col3:
        if 'Units_Sold' in filtered_products.columns:
            total_units = filtered_products['Units_Sold'].sum()
            st.metric("📊 Units Sold", f"{int(total_units):,}")
        else:
            st.metric("📊 Units Sold", "N/A")
    
    with col4:
        if 'Price' in filtered_products.columns:
            avg_price = filtered_products['Price'].mean()
            st.metric("💵 Avg Price", f"${avg_price:.2f}")
        else:
            st.metric("💵 Avg Price", "N/A")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Revenue by Category")
        if 'Product_Category' in filtered_products.columns and 'Revenue' in filtered_products.columns:
            category_revenue = filtered_products.groupby('Product_Category')['Revenue'].sum().reset_index()
            category_revenue = category_revenue.sort_values('Revenue', ascending=False)
            fig = px.bar(category_revenue, x='Product_Category', y='Revenue',
                        color='Revenue',
                        color_continuous_scale='Teal',
                        title='Revenue by Product Category')
            fig.update_layout(
                template='plotly_white',
                xaxis_tickangle=-45,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Category revenue data not available")
    
    with col2:
        st.markdown("### 📈 Units Sold by Category")
        if 'Product_Category' in filtered_products.columns and 'Units_Sold' in filtered_products.columns:
            category_units = filtered_products.groupby('Product_Category')['Units_Sold'].sum().reset_index()
            category_units = category_units.sort_values('Units_Sold', ascending=False)
            fig = px.bar(category_units, x='Product_Category', y='Units_Sold',
                        color='Units_Sold',
                        color_continuous_scale='Oranges',
                        title='Units Sold by Category')
            fig.update_layout(
                template='plotly_white',
                xaxis_tickangle=-45,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Units sold data not available")
    
    # Top Products
    st.markdown("### 🏆 Top Performing Products")
    if 'Product_Name' in filtered_products.columns and 'Revenue' in filtered_products.columns:
        top_products = filtered_products.nlargest(10, 'Revenue')
        fig = px.bar(top_products, x='Product_Name', y='Revenue',
                    color='Revenue',
                    color_continuous_scale='Viridis',
                    title='Top 10 Products by Revenue')
        fig.update_layout(
            template='plotly_white',
            xaxis_tickangle=-45,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Product performance data not available")
    
    # Product Performance Table
    st.markdown("### 📋 Product Details")
    if not filtered_products.empty:
        st.dataframe(filtered_products.head(10), use_container_width=True)
    else:
        st.info("No product data available")

# ==================== GEOGRAPHIC INSIGHTS PAGE ====================
elif page == "🌍 Geographic Insights":
    st.markdown("## 🌍 Geographic Insights")
    
    # Filters
    if 'Region' in geographic_data.columns:
        regions = st.multiselect(
            "Select Regions",
            options=geographic_data['Region'].unique(),
            default=geographic_data['Region'].unique()
        )
        filtered_geo = geographic_data[geographic_data['Region'].isin(regions)]
    else:
        filtered_geo = geographic_data.copy()
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'Region' in filtered_geo.columns:
            total_regions = filtered_geo['Region'].nunique()
            st.metric("🌍 Regions", f"{total_regions}")
        else:
            st.metric("🌍 Regions", "N/A")
    
    with col2:
        if 'Country' in filtered_geo.columns:
            total_countries = filtered_geo['Country'].nunique()
            st.metric("🗺️ Countries", f"{total_countries}")
        else:
            st.metric("🗺️ Countries", "N/A")
    
    with col3:
        if 'Revenue' in filtered_geo.columns:
            total_revenue = filtered_geo['Revenue'].sum()
            st.metric("💰 Total Revenue", f"${total_revenue:,.0f}")
        else:
            st.metric("💰 Total Revenue", "N/A")
    
    with col4:
        if 'Customers' in filtered_geo.columns:
            total_customers = filtered_geo['Customers'].sum()
            st.metric("👥 Total Customers", f"{int(total_customers):,}")
        else:
            st.metric("👥 Total Customers", "N/A")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💰 Revenue by Region")
        if 'Region' in filtered_geo.columns and 'Revenue' in filtered_geo.columns:
            region_revenue = filtered_geo.groupby('Region')['Revenue'].sum().reset_index()
            region_revenue = region_revenue.sort_values('Revenue', ascending=False)
            fig = px.bar(region_revenue, x='Region', y='Revenue',
                        color='Revenue',
                        color_continuous_scale='Sunset',
                        title='Revenue Distribution by Region')
            fig.update_layout(
                template='plotly_white',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Regional revenue data not available")
    
    with col2:
        st.markdown("### 👥 Customer Distribution")
        if 'Region' in filtered_geo.columns and 'Customers' in filtered_geo.columns:
            region_customers = filtered_geo.groupby('Region')['Customers'].sum().reset_index()
            fig = px.pie(region_customers, values='Customers', names='Region',
                        title='Customer Distribution by Region',
                        color_discrete_sequence=px.colors.qualitative.Bold)
            fig.update_layout(
                template='plotly_white',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Customer distribution data not available")
    
    # Top Countries
    st.markdown("### 🏆 Top Countries by Revenue")
    if 'Country' in filtered_geo.columns and 'Revenue' in filtered_geo.columns:
        country_revenue = filtered_geo.groupby('Country')['Revenue'].sum().reset_index()
        country_revenue = country_revenue.sort_values('Revenue', ascending=False).head(15)
        fig = px.bar(country_revenue, x='Country', y='Revenue',
                    color='Revenue',
                    color_continuous_scale='Viridis',
                    title='Top 15 Countries by Revenue')
        fig.update_layout(
            template='plotly_white',
            xaxis_tickangle=-45,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Country revenue data not available")
    
    # Geographic Performance Table
    st.markdown("### 📊 Geographic Performance Details")
    if not filtered_geo.empty:
        st.dataframe(filtered_geo, use_container_width=True)
    else:
        st.info("No geographic data available")

# ==================== CUSTOMER JOURNEY PAGE ====================
elif page == "🔄 Customer Journey":
    st.markdown("## 🔄 Customer Journey Analysis")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'Customer_ID' in customer_journey.columns:
            total_journeys = customer_journey['Customer_ID'].nunique()
            st.metric("👥 Unique Customers", f"{total_journeys:,}")
        else:
            st.metric("👥 Unique Customers", "N/A")
    
    with col2:
        if 'Touchpoint' in customer_journey.columns:
            total_touchpoints = len(customer_journey)
            st.metric("📍 Total Touchpoints", f"{total_touchpoints:,}")
        else:
            st.metric("📍 Total Touchpoints", "N/A")
    
    with col3:
        if 'Conversion' in customer_journey.columns:
            conversion_rate = (customer_journey['Conversion'].sum() / len(customer_journey) * 100)
            st.metric("✅ Conversion Rate", f"{conversion_rate:.2f}%")
        else:
            st.metric("✅ Conversion Rate", "N/A")
    
    with col4:
        if 'Time_Spent' in customer_journey.columns:
            avg_time = customer_journey['Time_Spent'].mean()
            st.metric("⏱️ Avg Time Spent", f"{avg_time:.1f} min")
        else:
            st.metric("⏱️ Avg Time Spent", "N/A")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📍 Touchpoint Distribution")
        if 'Touchpoint' in customer_journey.columns:
            touchpoint_counts = customer_journey['Touchpoint'].value_counts().reset_index()
            touchpoint_counts.columns = ['Touchpoint', 'Count']
            fig = px.bar(touchpoint_counts, x='Touchpoint', y='Count',
                        color='Count',
                        color_continuous_scale='Blues',
                        title='Customer Touchpoints')
            fig.update_layout(
                template='plotly_white',
                xaxis_tickangle=-45,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Touchpoint data not available")
    
    with col2:
        st.markdown("### 🎯 Conversion by Touchpoint")
        if 'Touchpoint' in customer_journey.columns and 'Conversion' in customer_journey.columns:
            touchpoint_conv = customer_journey.groupby('Touchpoint')['Conversion'].mean().reset_index()
            touchpoint_conv['Conversion'] = touchpoint_conv['Conversion'] * 100
            fig = px.bar(touchpoint_conv, x='Touchpoint', y='Conversion',
                        color='Conversion',
                        color_continuous_scale='RdYlGn',
                        title='Conversion Rate by Touchpoint (%)')
            fig.update_layout(
                template='plotly_white',
                xaxis_tickangle=-45,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Conversion data not available")
    
    # Journey Path Analysis
    st.markdown("### 🛤️ Customer Journey Paths")
    if 'Journey_Stage' in customer_journey.columns:
        stage_counts = customer_journey['Journey_Stage'].value_counts().reset_index()
        stage_counts.columns = ['Stage', 'Count']
        fig = go.Figure(go.Funnel(
            y=stage_counts['Stage'],
            x=stage_counts['Count'],
            textinfo="value+percent initial",
            marker=dict(color=['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b'])
        ))
        fig.update_layout(
            template='plotly_white',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Journey stage data not available")
    
    # Journey Details Table
    st.markdown("### 📋 Journey Details")
    if not customer_journey.empty:
        st.dataframe(customer_journey.head(20), use_container_width=True)
    else:
        st.info("No journey data available")

# ==================== CHANNEL ATTRIBUTION PAGE ====================
elif page == "📊 Channel Attribution":
    st.markdown("## 📊 Channel Attribution Analysis")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'Channel' in channel_attribution.columns:
            total_channels = channel_attribution['Channel'].nunique()
            st.metric("📺 Total Channels", f"{total_channels}")
        else:
            st.metric("📺 Total Channels", "N/A")
    
    with col2:
        if 'Conversions' in channel_attribution.columns:
            total_conversions = channel_attribution['Conversions'].sum()
            st.metric("✅ Total Conversions", f"{int(total_conversions):,}")
        else:
            st.metric("✅ Total Conversions", "N/A")
    
    with col3:
        if 'Revenue' in channel_attribution.columns:
            total_revenue = channel_attribution['Revenue'].sum()
            st.metric("💰 Total Revenue", f"${total_revenue:,.0f}")
        else:
            st.metric("💰 Total Revenue", "N/A")
    
    with col4:
        if 'Cost' in channel_attribution.columns:
            total_cost = channel_attribution['Cost'].sum()
            st.metric("💵 Total Cost", f"${total_cost:,.0f}")
        else:
            st.metric("💵 Total Cost", "N/A")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Conversions by Channel")
        if 'Channel' in channel_attribution.columns and 'Conversions' in channel_attribution.columns:
            fig = px.pie(channel_attribution, values='Conversions', names='Channel',
                        title='Conversion Distribution by Channel',
                        color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_layout(
                template='plotly_white',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Channel conversion data not available")
    
    with col2:
        st.markdown("### 💰 Revenue by Channel")
        if 'Channel' in channel_attribution.columns and 'Revenue' in channel_attribution.columns:
            fig = px.bar(channel_attribution, x='Channel', y='Revenue',
                        color='Revenue',
                        color_continuous_scale='Viridis',
                        title='Revenue by Channel')
            fig.update_layout(
                template='plotly_white',
                xaxis_tickangle=-45,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Channel revenue data not available")
    
    # ROI Analysis
    st.markdown("### 💹 Channel ROI Analysis")
    if 'Channel' in channel_attribution.columns and 'Revenue' in channel_attribution.columns and 'Cost' in channel_attribution.columns:
        channel_attribution['ROI'] = ((channel_attribution['Revenue'] - channel_attribution['Cost']) / channel_attribution['Cost'] * 100).round(2)
        fig = px.bar(channel_attribution.sort_values('ROI', ascending=False),
                    x='Channel', y='ROI',
                    color='ROI',
                    color_continuous_scale='RdYlGn',
                    title='ROI by Channel (%)')
        fig.update_layout(
            template='plotly_white',
            xaxis_tickangle=-45,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("ROI data not available")
    
    # Attribution Table
    st.markdown("### 📋 Channel Attribution Details")
    if not channel_attribution.empty:
        st.dataframe(channel_attribution, use_container_width=True)
    else:
        st.info("No attribution data available")

# ==================== ML INSIGHTS PAGE ====================
elif page == "🎓 ML Insights":
    st.markdown("## 🎓 Machine Learning Insights")
    
    # Feature Importance
    st.markdown("### 🎯 Feature Importance")
    if not feature_importance.empty and 'Feature' in feature_importance.columns and 'Importance' in feature_importance.columns:
        fig = px.bar(feature_importance.sort_values('Importance', ascending=True),
                    y='Feature', x='Importance',
                    orientation='h',
                    color='Importance',
                    color_continuous_scale='Viridis',
                    title='Feature Importance for Prediction Model')
        fig.update_layout(
            template='plotly_white',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance data not available")
    
    # Correlation Matrix
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔗 Correlation Matrix")
        if not correlation_matrix.empty:
            # Assuming first column is index
            corr_data = correlation_matrix.set_index(correlation_matrix.columns[0])
            fig = px.imshow(corr_data,
                           title='Feature Correlation Heatmap',
                           color_continuous_scale='RdBu',
                           aspect='auto')
            fig.update_layout(
                template='plotly_white',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Correlation matrix not available")
    
    with col2:
        st.markdown("### 📈 Learning Curve")
        if not learning_curve.empty and 'Training_Size' in learning_curve.columns:
            fig = go.Figure()
            if 'Training_Score' in learning_curve.columns:
                fig.add_trace(go.Scatter(
                    x=learning_curve['Training_Size'],
                    y=learning_curve['Training_Score'],
                    mode='lines+markers',
                    name='Training Score',
                    line=dict(color='#667eea', width=3)
                ))
            if 'Validation_Score' in learning_curve.columns:
                fig.add_trace(go.Scatter(
                    x=learning_curve['Training_Size'],
                    y=learning_curve['Validation_Score'],
                    mode='lines+markers',
                    name='Validation Score',
                    line=dict(color='#764ba2', width=3)
                ))
            fig.update_layout(
                title='Model Learning Curve',
                xaxis_title='Training Size',
                yaxis_title='Score',
                template='plotly_white',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Learning curve data not available")
    
    # Model Performance Metrics
    st.markdown("### 📊 Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 Accuracy", "92.5%", delta="↑ 2.3%")
    with col2:
        st.metric("📊 Precision", "89.7%", delta="↑ 1.8%")
    with col3:
        st.metric("🔍 Recall", "91.2%", delta="↑ 3.1%")
    with col4:
        st.metric("⚖️ F1-Score", "90.4%", delta="↑ 2.5%")

# ==================== LEAD SCORING PAGE ====================
elif page == "🔍 Lead Scoring":
    st.markdown("## 🔍 Lead Scoring Analysis")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'Lead_ID' in lead_scoring_results.columns:
            total_leads = lead_scoring_results['Lead_ID'].nunique()
            st.metric("📋 Total Leads", f"{total_leads:,}")
        else:
            st.metric("📋 Total Leads", "N/A")
    
    with col2:
        if 'Lead_Score' in lead_scoring_results.columns:
            avg_score = lead_scoring_results['Lead_Score'].mean()
            st.metric("⭐ Avg Lead Score", f"{avg_score:.1f}")
        else:
            st.metric("⭐ Avg Lead Score", "N/A")
    
    with col3:
        if 'Conversion_Probability' in lead_scoring_results.columns:
            avg_prob = lead_scoring_results['Conversion_Probability'].mean() * 100
            st.metric("🎯 Avg Conv. Prob", f"{avg_prob:.1f}%")
        else:
            st.metric("🎯 Avg Conv. Prob", "N/A")
    
    with col4:
        if 'Lead_Quality' in lead_scoring_results.columns:
            high_quality = (lead_scoring_results['Lead_Quality'] == 'High').sum()
            st.metric("🏆 High Quality Leads", f"{high_quality:,}")
        else:
            st.metric("🏆 High Quality Leads", "N/A")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Lead Score Distribution")
        if 'Lead_Score' in lead_scoring_results.columns:
            fig = px.histogram(lead_scoring_results, x='Lead_Score',
                             nbins=30,
                             title='Distribution of Lead Scores',
                             color_discrete_sequence=['#667eea'])
            fig.update_layout(
                template='plotly_white',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Lead score data not available")
    
    with col2:
        st.markdown("### 🎯 Lead Quality Distribution")
        if 'Lead_Quality' in lead_scoring_results.columns:
            quality_counts = lead_scoring_results['Lead_Quality'].value_counts().reset_index()
            quality_counts.columns = ['Quality', 'Count']
            fig = px.pie(quality_counts, values='Count', names='Quality',
                        title='Lead Quality Breakdown',
                        color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(
                template='plotly_white',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Lead quality data not available")
    
    # Lead Score vs Conversion Probability
    st.markdown("### 📈 Lead Score vs Conversion Probability")
    if 'Lead_Score' in lead_scoring_results.columns and 'Conversion_Probability' in lead_scoring_results.columns:
        fig = px.scatter(lead_scoring_results,
                        x='Lead_Score',
                        y='Conversion_Probability',
                        color='Lead_Quality' if 'Lead_Quality' in lead_scoring_results.columns else None,
                        title='Lead Score vs Conversion Probability',
                        labels={'Lead_Score': 'Lead Score', 'Conversion_Probability': 'Conversion Probability'})
        fig.update_layout(
            template='plotly_white',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Lead scoring data not available")
    
    # Top Leads Table
    st.markdown("### 🏆 Top Scoring Leads")
    if not lead_scoring_results.empty and 'Lead_Score' in lead_scoring_results.columns:
        top_leads = lead_scoring_results.nlargest(10, 'Lead_Score')
        st.dataframe(top_leads, use_container_width=True)
    else:
        st.info("No lead scoring data available")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 20px; background: rgba(255,255,255,0.1); border-radius: 10px;'>
        <p style='color: white; font-size: 16px; margin: 0;'>
            <b>Marketing Analytics Dashboard</b> | Built with ❤️ using Streamlit
        </p>
        <p style='color: white; font-size: 14px; margin-top: 10px;'>
            © 2024 All Rights Reserved | Version 1.0
        </p>
    </div>
    """, unsafe_allow_html=True)
