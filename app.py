# Add after imports
import warnings
warnings.filterwarnings('ignore')

# Reduce plotly file size
import plotly.io as pio
pio.templates.default = "plotly_white"

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime

# Page configuration - MUST BE FIRST
st.set_page_config(
    page_title="Marketing Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Optimized CSS - Simplified for faster loading
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
    }
    .stMetric {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1, h2 {
        color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

# Optimized data loading with better caching
@st.cache_data(ttl=3600, show_spinner=False)
def load_all_data():
    """Load all CSV files at once - cached for 1 hour"""
    data_dict = {}
    file_names = [
        'campaign_performance', 'channel_attribution', 'correlation_matrix',
        'customer_data', 'customer_journey', 'feature_importance',
        'funnel_data', 'geographic_data', 'lead_scoring_results',
        'learning_curve', 'product_sales'
    ]
    
    for file_name in file_names:
        try:
            data_dict[file_name] = pd.read_csv(f'{file_name}.csv')
        except FileNotFoundError:
            st.warning(f"⚠️ {file_name}.csv not found")
            data_dict[file_name] = pd.DataFrame()
    
    return data_dict

# Load data with spinner
with st.spinner('🚀 Loading data...'):
    data = load_all_data()

# Extract datasets
campaign_performance = data['campaign_performance']
channel_attribution = data['channel_attribution']
correlation_matrix = data['correlation_matrix']
customer_data = data['customer_data']
customer_journey = data['customer_journey']
feature_importance = data['feature_importance']
funnel_data = data['funnel_data']
geographic_data = data['geographic_data']
lead_scoring_results = data['lead_scoring_results']
learning_curve = data['learning_curve']
product_sales = data['product_sales']

# Sidebar Navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Select Dashboard",
    ["🏠 Overview", "📈 Campaign Performance", "🎯 Customer Analytics", 
     "🛍️ Product Sales", "🌍 Geographic Insights", "🔄 Customer Journey",
     "📊 Channel Attribution", "🎓 ML Insights", "🔍 Lead Scoring"],
    key="nav_radio"
)

st.sidebar.markdown("---")
st.sidebar.info("**Marketing Analytics Dashboard**\n\nData-Driven Insights")

# Main title
st.title("🎯 Marketing Analytics Dashboard")

# ==================== OVERVIEW PAGE ====================
if page == "🏠 Overview":
    st.header("📊 Executive Summary")
    
    # Key Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("👥 Customers", f"{len(customer_data):,}")
    
    with col2:
        if 'Revenue' in product_sales.columns:
            st.metric("💰 Revenue", f"${product_sales['Revenue'].sum():,.0f}")
        else:
            st.metric("💰 Revenue", "N/A")
    
    with col3:
        if 'Conversions' in campaign_performance.columns:
            st.metric("✅ Conversions", f"{int(campaign_performance['Conversions'].sum()):,}")
        else:
            st.metric("✅ Conversions", "N/A")
    
    with col4:
        if 'CTR' in campaign_performance.columns:
            st.metric("📊 Avg CTR", f"{campaign_performance['CTR'].mean():.2f}%")
        else:
            st.metric("📊 Avg CTR", "N/A")
    
    with col5:
        if 'ROI' in campaign_performance.columns:
            st.metric("💹 Avg ROI", f"{campaign_performance['ROI'].mean():.1f}%")
        else:
            st.metric("💹 Avg ROI", "N/A")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Revenue Trend")
        if 'Date' in campaign_performance.columns and 'Revenue' in campaign_performance.columns:
            campaign_performance['Date'] = pd.to_datetime(campaign_performance['Date'], errors='coerce')
            daily_rev = campaign_performance.groupby('Date')['Revenue'].sum().reset_index()
            fig = px.line(daily_rev, x='Date', y='Revenue', template='plotly_white')
            fig.update_traces(line_color='#667eea', line_width=2)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("Revenue data not available")
    
    with col2:
        st.subheader("🎯 Conversion Funnel")
        if 'Stage' in funnel_data.columns and 'Count' in funnel_data.columns:
            fig = go.Figure(go.Funnel(
                y=funnel_data['Stage'],
                x=funnel_data['Count'],
                textinfo="value+percent initial"
            ))
            fig.update_layout(template='plotly_white')
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("Funnel data not available")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌍 Geographic Revenue")
        if 'Region' in geographic_data.columns and 'Revenue' in geographic_data.columns:
            fig = px.bar(geographic_data, x='Region', y='Revenue', color='Revenue',
                        color_continuous_scale='Viridis', template='plotly_white')
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("Geographic data not available")
    
    with col2:
        st.subheader("📊 Channel Performance")
        if 'Channel' in channel_attribution.columns and 'Conversions' in channel_attribution.columns:
            fig = px.pie(channel_attribution, values='Conversions', names='Channel',
                        color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("Channel data not available")

# ==================== CAMPAIGN PERFORMANCE PAGE ====================
elif page == "📈 Campaign Performance":
    st.header("📈 Campaign Performance")
    
    # Filters
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Campaign_Name' in campaign_performance.columns:
            campaigns = st.multiselect(
                "Select Campaigns",
                options=campaign_performance['Campaign_Name'].unique().tolist()[:10],
                default=campaign_performance['Campaign_Name'].unique().tolist()[:3],
                key="camp_filter"
            )
            if campaigns:
                filtered_cp = campaign_performance[campaign_performance['Campaign_Name'].isin(campaigns)]
            else:
                filtered_cp = campaign_performance.head(100)
        else:
            filtered_cp = campaign_performance.head(100)
    
    with col2:
        if 'Channel' in campaign_performance.columns:
            channels = st.multiselect(
                "Select Channels",
                options=campaign_performance['Channel'].unique().tolist(),
                default=campaign_performance['Channel'].unique().tolist()[:3],
                key="channel_filter"
            )
            if channels:
                filtered_cp = filtered_cp[filtered_cp['Channel'].isin(channels)]
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'Impressions' in filtered_cp.columns:
            st.metric("👁️ Impressions", f"{filtered_cp['Impressions'].sum():,.0f}")
    
    with col2:
        if 'Clicks' in filtered_cp.columns:
            st.metric("🖱️ Clicks", f"{int(filtered_cp['Clicks'].sum()):,}")
    
    with col3:
        if 'Conversions' in filtered_cp.columns:
            st.metric("✅ Conversions", f"{int(filtered_cp['Conversions'].sum()):,}")
    
    with col4:
        if 'Spend' in filtered_cp.columns:
            st.metric("💵 Spend", f"${filtered_cp['Spend'].sum():,.2f}")
    
    st.markdown("---")
    
    # Table
    st.subheader("📊 Campaign Details")
    st.dataframe(filtered_cp.head(10), use_container_width=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Campaign_Name' in filtered_cp.columns and 'CTR' in filtered_cp.columns:
            top_ctr = filtered_cp.nlargest(10, 'CTR')
            fig = px.bar(top_ctr, x='Campaign_Name', y='CTR', color='CTR',
                        color_continuous_scale='Blues', template='plotly_white')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    with col2:
        if 'Campaign_Name' in filtered_cp.columns and 'ROI' in filtered_cp.columns:
            top_roi = filtered_cp.nlargest(10, 'ROI')
            fig = px.bar(top_roi, x='Campaign_Name', y='ROI', color='ROI',
                        color_continuous_scale='RdYlGn', template='plotly_white')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# ==================== CUSTOMER ANALYTICS PAGE ====================
elif page == "🎯 Customer Analytics":
    st.header("🎯 Customer Analytics")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'Age_Group' in customer_data.columns:
            age_groups = st.multiselect(
                "Age Group",
                options=customer_data['Age_Group'].unique().tolist(),
                default=customer_data['Age_Group'].unique().tolist(),
                key="age_filter"
            )
            filtered_customers = customer_data[customer_data['Age_Group'].isin(age_groups)] if age_groups else customer_data
        else:
            filtered_customers = customer_data
    
    with col2:
        if 'Income_Level' in customer_data.columns:
            income_levels = st.multiselect(
                "Income Level",
                options=customer_data['Income_Level'].unique().tolist(),
                default=customer_data['Income_Level'].unique().tolist(),
                key="income_filter"
            )
            if income_levels:
                filtered_customers = filtered_customers[filtered_customers['Income_Level'].isin(income_levels)]
    
    with col3:
        if 'Customer_Segment' in customer_data.columns:
            segments = st.multiselect(
                "Segment",
                options=customer_data['Customer_Segment'].unique().tolist(),
                default=customer_data['Customer_Segment'].unique().tolist(),
                key="segment_filter"
            )
            if segments:
                filtered_customers = filtered_customers[filtered_customers['Customer_Segment'].isin(segments)]
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("👥 Customers", f"{len(filtered_customers):,}")
    
    with col2:
        if 'Total_Spend' in filtered_customers.columns:
            st.metric("💰 Avg Value", f"${filtered_customers['Total_Spend'].mean():,.2f}")
    
    with col3:
        if 'Purchase_Frequency' in filtered_customers.columns:
            st.metric("🔄 Avg Frequency", f"{filtered_customers['Purchase_Frequency'].mean():.1f}")
    
    with col4:
        if 'Churn_Risk' in filtered_customers.columns:
            churn = (filtered_customers['Churn_Risk'] == 'High').sum() / len(filtered_customers) * 100
            st.metric("⚠️ Churn Risk", f"{churn:.1f}%")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Customer_Segment' in filtered_customers.columns:
            segment_counts = filtered_customers['Customer_Segment'].value_counts().reset_index()
            segment_counts.columns = ['Segment', 'Count']
            fig = px.pie(segment_counts, values='Count', names='Segment',
                        color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    with col2:
        if 'Age_Group' in filtered_customers.columns and 'Total_Spend' in filtered_customers.columns:
            age_revenue = filtered_customers.groupby('Age_Group')['Total_Spend'].sum().reset_index()
            fig = px.bar(age_revenue, x='Age_Group', y='Total_Spend', color='Total_Spend',
                        color_continuous_scale='Viridis', template='plotly_white')
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    st.dataframe(filtered_customers.head(10), use_container_width=True)

# ==================== PRODUCT SALES PAGE ====================
elif page == "🛍️ Product Sales":
    st.header("🛍️ Product Sales")
    
    # Filters
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Product_Category' in product_sales.columns:
            categories = st.multiselect(
                "Category",
                options=product_sales['Product_Category'].unique().tolist(),
                default=product_sales['Product_Category'].unique().tolist(),
                key="prod_cat_filter"
            )
            filtered_products = product_sales[product_sales['Product_Category'].isin(categories)] if categories else product_sales
        else:
            filtered_products = product_sales
    
    with col2:
        if 'Price' in product_sales.columns:
            price_range = st.slider(
                "Price Range",
                min_value=float(product_sales['Price'].min()),
                max_value=float(product_sales['Price'].max()),
                value=(float(product_sales['Price'].min()), float(product_sales['Price'].max())),
                key="price_slider"
            )
            filtered_products = filtered_products[
                (filtered_products['Price'] >= price_range[0]) & 
                (filtered_products['Price'] <= price_range[1])
            ]
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'Product_ID' in filtered_products.columns:
            st.metric("📦 Products", f"{filtered_products['Product_ID'].nunique():,}")
    
    with col2:
        if 'Revenue' in filtered_products.columns:
            st.metric("💰 Revenue", f"${filtered_products['Revenue'].sum():,.0f}")
    
    with col3:
        if 'Units_Sold' in filtered_products.columns:
            st.metric("📊 Units Sold", f"{int(filtered_products['Units_Sold'].sum()):,}")
    
    with col4:
        if 'Price' in filtered_products.columns:
            st.metric("💵 Avg Price", f"${filtered_products['Price'].mean():.2f}")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Product_Category' in filtered_products.columns and 'Revenue' in filtered_products.columns:
            cat_rev = filtered_products.groupby('Product_Category')['Revenue'].sum().reset_index()
            fig = px.bar(cat_rev, x='Product_Category', y='Revenue', color='Revenue',
                        color_continuous_scale='Teal', template='plotly_white')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    with col2:
        if 'Product_Category' in filtered_products.columns and 'Units_Sold' in filtered_products.columns:
            cat_units = filtered_products.groupby('Product_Category')['Units_Sold'].sum().reset_index()
            fig = px.bar(cat_units, x='Product_Category', y='Units_Sold', color='Units_Sold',
                        color_continuous_scale='Oranges', template='plotly_white')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    st.dataframe(filtered_products.head(10), use_container_width=True)

# ==================== GEOGRAPHIC INSIGHTS PAGE ====================
elif page == "🌍 Geographic Insights":
    st.header("🌍 Geographic Insights")
    
    if 'Region' in geographic_data.columns:
        regions = st.multiselect(
            "Select Regions",
            options=geographic_data['Region'].unique().tolist(),
            default=geographic_data['Region'].unique().tolist(),
            key="region_filter"
        )
        filtered_geo = geographic_data[geographic_data['Region'].isin(regions)] if regions else geographic_data
    else:
        filtered_geo = geographic_data
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'Region' in filtered_geo.columns:
            st.metric("🌍 Regions", f"{filtered_geo['Region'].nunique()}")
    
    with col2:
        if 'Country' in filtered_geo.columns:
            st.metric("🗺️ Countries", f"{filtered_geo['Country'].nunique()}")
    
    with col3:
        if 'Revenue' in filtered_geo.columns:
            st.metric("💰 Revenue", f"${filtered_geo['Revenue'].sum():,.0f}")
    
    with col4:
        if 'Customers' in filtered_geo.columns:
            st.metric("👥 Customers", f"{int(filtered_geo['Customers'].sum()):,}")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Region' in filtered_geo.columns and 'Revenue' in filtered_geo.columns:
            region_rev = filtered_geo.groupby('Region')['Revenue'].sum().reset_index()
            fig = px.bar(region_rev, x='Region', y='Revenue', color='Revenue',
                        color_continuous_scale='Sunset', template='plotly_white')
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    with col2:
        if 'Region' in filtered_geo.columns and 'Customers' in filtered_geo.columns:
            region_cust = filtered_geo.groupby('Region')['Customers'].sum().reset_index()
            fig = px.pie(region_cust, values='Customers', names='Region',
                        color_discrete_sequence=px.colors.qualitative.Bold)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    st.dataframe(filtered_geo.head(10), use_container_width=True)

# ==================== CUSTOMER JOURNEY PAGE ====================
elif page == "🔄 Customer Journey":
    st.header("🔄 Customer Journey")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'Customer_ID' in customer_journey.columns:
            st.metric("👥 Customers", f"{customer_journey['Customer_ID'].nunique():,}")
    
    with col2:
        if 'Touchpoint' in customer_journey.columns:
            st.metric("📍 Touchpoints", f"{len(customer_journey):,}")
    
    with col3:
        if 'Conversion' in customer_journey.columns:
            conv_rate = (customer_journey['Conversion'].sum() / len(customer_journey) * 100)
            st.metric("✅ Conv. Rate", f"{conv_rate:.2f}%")
    
    with col4:
        if 'Time_Spent' in customer_journey.columns:
            st.metric("⏱️ Avg Time", f"{customer_journey['Time_Spent'].mean():.1f} min")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Touchpoint' in customer_journey.columns:
            tp_counts = customer_journey['Touchpoint'].value_counts().reset_index()
            tp_counts.columns = ['Touchpoint', 'Count']
            fig = px.bar(tp_counts, x='Touchpoint', y='Count', color='Count',
                        color_continuous_scale='Blues', template='plotly_white')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    with col2:
        if 'Touchpoint' in customer_journey.columns and 'Conversion' in customer_journey.columns:
            tp_conv = customer_journey.groupby('Touchpoint')['Conversion'].mean().reset_index()
            tp_conv['Conversion'] = tp_conv['Conversion'] * 100
            fig = px.bar(tp_conv, x='Touchpoint', y='Conversion', color='Conversion',
                        color_continuous_scale='RdYlGn', template='plotly_white')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    st.dataframe(customer_journey.head(10), use_container_width=True)

# ==================== CHANNEL ATTRIBUTION PAGE ====================
elif page == "📊 Channel Attribution":
    st.header("📊 Channel Attribution")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'Channel' in channel_attribution.columns:
            st.metric("📺 Channels", f"{channel_attribution['Channel'].nunique()}")
    
    with col2:
        if 'Conversions' in channel_attribution.columns:
            st.metric("✅ Conversions", f"{int(channel_attribution['Conversions'].sum()):,}")
    
    with col3:
        if 'Revenue' in channel_attribution.columns:
            st.metric("💰 Revenue", f"${channel_attribution['Revenue'].sum():,.0f}")
    
    with col4:
        if 'Cost' in channel_attribution.columns:
            st.metric("💵 Cost", f"${channel_attribution['Cost'].sum():,.0f}")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Channel' in channel_attribution.columns and 'Conversions' in channel_attribution.columns:
            fig = px.pie(channel_attribution, values='Conversions', names='Channel',
                        color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    with col2:
        if 'Channel' in channel_attribution.columns and 'Revenue' in channel_attribution.columns:
            fig = px.bar(channel_attribution, x='Channel', y='Revenue', color='Revenue',
                        color_continuous_scale='Viridis', template='plotly_white')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    st.dataframe(channel_attribution, use_container_width=True)

# ==================== ML INSIGHTS PAGE ====================
elif page == "🎓 ML Insights":
    st.header("🎓 ML Insights")
    
    # Feature Importance
    if not feature_importance.empty and 'Feature' in feature_importance.columns and 'Importance' in feature_importance.columns:
        fig = px.bar(feature_importance.sort_values('Importance', ascending=True),
                    y='Feature', x='Importance', orientation='h', color='Importance',
                    color_continuous_scale='Viridis', template='plotly_white')
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    col1, col2 = st.columns(2)
    
    with col1:
        if not correlation_matrix.empty:
            corr_data = correlation_matrix.set_index(correlation_matrix.columns[0])
            fig = px.imshow(corr_data, color_continuous_scale='RdBu', aspect='auto')
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    with col2:
        if not learning_curve.empty and 'Training_Size' in learning_curve.columns:
            fig = go.Figure()
            if 'Training_Score' in learning_curve.columns:
                fig.add_trace(go.Scatter(x=learning_curve['Training_Size'],
                                        y=learning_curve['Training_Score'],
                                        mode='lines+markers', name='Training'))
            if 'Validation_Score' in learning_curve.columns:
                fig.add_trace(go.Scatter(x=learning_curve['Training_Size'],
                                        y=learning_curve['Validation_Score'],
                                        mode='lines+markers', name='Validation'))
            fig.update_layout(template='plotly_white')
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# ==================== LEAD SCORING PAGE ====================
elif page == "🔍 Lead Scoring":
    st.header("🔍 Lead Scoring")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'Lead_ID' in lead_scoring_results.columns:
            st.metric("📋 Leads", f"{lead_scoring_results['Lead_ID'].nunique():,}")
    
    with col2:
        if 'Lead_Score' in lead_scoring_results.columns:
            st.metric("⭐ Avg Score", f"{lead_scoring_results['Lead_Score'].mean():.1f}")
    
    with col3:
        if 'Conversion_Probability' in lead_scoring_results.columns:
            st.metric("🎯 Avg Prob", f"{lead_scoring_results['Conversion_Probability'].mean() * 100:.1f}%")
    
    with col4:
        if 'Lead_Quality' in lead_scoring_results.columns:
            high_quality = (lead_scoring_results['Lead_Quality'] == 'High').sum()
            st.metric("🏆 High Quality", f"{high_quality:,}")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Lead_Score' in lead_scoring_results.columns:
            fig = px.histogram(lead_scoring_results, x='Lead_Score', nbins=30,
                             color_discrete_sequence=['#667eea'], template='plotly_white')
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    with col2:
        if 'Lead_Quality' in lead_scoring_results.columns:
            quality_counts = lead_scoring_results['Lead_Quality'].value_counts().reset_index()
            quality_counts.columns = ['Quality', 'Count']
            fig = px.pie(quality_counts, values='Count', names='Quality',
                        color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    st.dataframe(lead_scoring_results.head(10), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: white;'>Marketing Analytics Dashboard | Built with Streamlit</p>", 
            unsafe_allow_html=True)

