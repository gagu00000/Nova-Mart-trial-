import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import os

# Page configuration
st.set_page_config(
    page_title="Marketing Campaign Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    h1 {
        color: #1f77b4;
    }
    h2 {
        color: #2c3e50;
    }
    </style>
    """, unsafe_allow_html=True)

# Load data with caching
@st.cache_data
def load_data():
    """Load all CSV files"""
    try:
        # Try to load from current directory first
        if os.path.exists('customers.csv'):
            customers = pd.read_csv('customers.csv')
            products = pd.read_csv('products.csv')
            engagement = pd.read_csv('engagement_data.csv')
            geography = pd.read_csv('geography.csv')
        # Try data folder
        elif os.path.exists('data/customers.csv'):
            customers = pd.read_csv('data/customers.csv')
            products = pd.read_csv('data/products.csv')
            engagement = pd.read_csv('data/engagement_data.csv')
            geography = pd.read_csv('data/geography.csv')
        else:
            return None, None, None, None
        
        # Convert date columns
        if 'engagement_date' in engagement.columns:
            engagement['engagement_date'] = pd.to_datetime(engagement['engagement_date'])
        
        return customers, products, engagement, geography
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None

# Check if files exist, if not show upload interface
def check_and_upload_files():
    """Check if CSV files exist, otherwise provide upload interface"""
    
    files_exist = os.path.exists('customers.csv') or os.path.exists('data/customers.csv')
    
    if not files_exist:
        st.warning("⚠️ CSV files not found. Please upload your data files.")
        
        st.markdown("### 📁 Upload Your Data Files")
        st.info("Please upload all four CSV files: customers.csv, products.csv, engagement_data.csv, and geography.csv")
        
        col1, col2 = st.columns(2)
        
        with col1:
            customers_file = st.file_uploader("Upload customers.csv", type=['csv'], key='customers')
            products_file = st.file_uploader("Upload products.csv", type=['csv'], key='products')
        
        with col2:
            engagement_file = st.file_uploader("Upload engagement_data.csv", type=['csv'], key='engagement')
            geography_file = st.file_uploader("Upload geography.csv", type=['csv'], key='geography')
        
        if customers_file and products_file and engagement_file and geography_file:
            try:
                customers = pd.read_csv(customers_file)
                products = pd.read_csv(products_file)
                engagement = pd.read_csv(engagement_file)
                geography = pd.read_csv(geography_file)
                
                # Convert date columns
                if 'engagement_date' in engagement.columns:
                    engagement['engagement_date'] = pd.to_datetime(engagement['engagement_date'])
                
                st.success("✅ All files uploaded successfully!")
                return customers, products, engagement, geography
            except Exception as e:
                st.error(f"Error reading uploaded files: {e}")
                return None, None, None, None
        else:
            st.info("👆 Please upload all four CSV files to continue.")
            return None, None, None, None
    
    return None, None, None, None

# Try to load data from files
customers, products, engagement, geography = load_data()

# If files don't exist, show upload interface
if customers is None:
    uploaded_data = check_and_upload_files()
    if uploaded_data[0] is not None:
        customers, products, engagement, geography = uploaded_data
    else:
        st.stop()

# Sidebar
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Select a Page",
    ["Overview", "Customer Analysis", "Product Performance", "Engagement Insights", "Geographic Analysis", "Campaign ROI"]
)

st.sidebar.markdown("---")
st.sidebar.info("**Marketing Campaign Dashboard**\n\nAnalyze customer behavior, product performance, and campaign effectiveness.")

# Main title
st.title("🎯 Marketing Campaign Analytics Dashboard")

# Merge datasets for comprehensive analysis
@st.cache_data
def merge_data(customers, products, engagement, geography):
    """Merge all datasets for analysis"""
    # Merge engagement with customers
    df = engagement.merge(customers, on='customer_id', how='left')
    # Merge with products
    df = df.merge(products, on='product_id', how='left')
    # Merge with geography
    df = df.merge(geography, on='customer_id', how='left')
    return df

df_merged = merge_data(customers, products, engagement, geography)

# ==================== OVERVIEW PAGE ====================
if page == "Overview":
    st.header("📈 Executive Summary")
    
    # Key Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_customers = customers['customer_id'].nunique()
        st.metric("Total Customers", f"{total_customers:,}")
    
    with col2:
        total_revenue = df_merged['revenue'].sum()
        st.metric("Total Revenue", f"${total_revenue:,.2f}")
    
    with col3:
        total_products = products['product_id'].nunique()
        st.metric("Total Products", f"{total_products}")
    
    with col4:
        avg_conversion = (df_merged['conversion'].sum() / len(df_merged)) * 100
        st.metric("Avg Conversion Rate", f"{avg_conversion:.2f}%")
    
    with col5:
        total_engagement = len(engagement)
        st.metric("Total Engagements", f"{total_engagement:,}")
    
    st.markdown("---")
    
    # Two column layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Revenue Trend Over Time")
        revenue_trend = df_merged.groupby(df_merged['engagement_date'].dt.to_period('M'))['revenue'].sum().reset_index()
        revenue_trend['engagement_date'] = revenue_trend['engagement_date'].astype(str)
        
        fig_revenue = px.line(
            revenue_trend,
            x='engagement_date',
            y='revenue',
            title='Monthly Revenue Trend',
            labels={'engagement_date': 'Month', 'revenue': 'Revenue ($)'}
        )
        fig_revenue.update_traces(line_color='#1f77b4', line_width=3)
        st.plotly_chart(fig_revenue, use_container_width=True)
    
    with col2:
        st.subheader("Customer Segmentation")
        segment_counts = customers['income_level'].value_counts().reset_index()
        segment_counts.columns = ['income_level', 'count']
        
        fig_segment = px.pie(
            segment_counts,
            values='count',
            names='income_level',
            title='Customers by Income Level',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_segment, use_container_width=True)
    
    # Full width chart
    st.subheader("Channel Performance")
    channel_performance = df_merged.groupby('channel').agg({
        'revenue': 'sum',
        'conversion': 'sum',
        'customer_id': 'count'
    }).reset_index()
    channel_performance.columns = ['channel', 'revenue', 'conversions', 'engagements']
    
    fig_channel = go.Figure()
    fig_channel.add_trace(go.Bar(
        x=channel_performance['channel'],
        y=channel_performance['revenue'],
        name='Revenue',
        marker_color='#1f77b4'
    ))
    fig_channel.add_trace(go.Bar(
        x=channel_performance['channel'],
        y=channel_performance['conversions'] * 1000,  # Scale for visibility
        name='Conversions (x1000)',
        marker_color='#ff7f0e'
    ))
    fig_channel.update_layout(
        title='Revenue and Conversions by Channel',
        xaxis_title='Channel',
        yaxis_title='Value',
        barmode='group'
    )
    st.plotly_chart(fig_channel, use_container_width=True)

# ==================== CUSTOMER ANALYSIS PAGE ====================
elif page == "Customer Analysis":
    st.header("👥 Customer Analysis")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        income_filter = st.multiselect(
            "Income Level",
            options=customers['income_level'].unique(),
            default=customers['income_level'].unique()
        )
    with col2:
        age_filter = st.multiselect(
            "Age Group",
            options=customers['age_group'].unique(),
            default=customers['age_group'].unique()
        )
    with col3:
        loyalty_filter = st.multiselect(
            "Loyalty Status",
            options=customers['loyalty_status'].unique(),
            default=customers['loyalty_status'].unique()
        )
    
    # Filter data
    filtered_customers = customers[
        (customers['income_level'].isin(income_filter)) &
        (customers['age_group'].isin(age_filter)) &
        (customers['loyalty_status'].isin(loyalty_filter))
    ]
    
    filtered_df = df_merged[df_merged['customer_id'].isin(filtered_customers['customer_id'])]
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Filtered Customers", f"{filtered_customers['customer_id'].nunique():,}")
    with col2:
        avg_revenue = filtered_df.groupby('customer_id')['revenue'].sum().mean()
        st.metric("Avg Revenue/Customer", f"${avg_revenue:.2f}")
    with col3:
        loyalty_rate = (filtered_customers['loyalty_status'] == 'Loyal').sum() / len(filtered_customers) * 100 if len(filtered_customers) > 0 else 0
        st.metric("Loyalty Rate", f"{loyalty_rate:.1f}%")
    with col4:
        avg_age = filtered_customers['age'].mean()
        st.metric("Average Age", f"{avg_age:.1f}")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Age Distribution")
        fig_age = px.histogram(
            filtered_customers,
            x='age',
            nbins=20,
            title='Customer Age Distribution',
            labels={'age': 'Age', 'count': 'Number of Customers'},
            color_discrete_sequence=['#1f77b4']
        )
        st.plotly_chart(fig_age, use_container_width=True)
    
    with col2:
        st.subheader("Revenue by Age Group")
        age_revenue = filtered_df.groupby('age_group')['revenue'].sum().reset_index()
        fig_age_rev = px.bar(
            age_revenue,
            x='age_group',
            y='revenue',
            title='Total Revenue by Age Group',
            labels={'age_group': 'Age Group', 'revenue': 'Revenue ($)'},
            color='revenue',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_age_rev, use_container_width=True)
    
    # Customer Lifetime Value Analysis
    st.subheader("Customer Lifetime Value Analysis")
    clv_data = filtered_df.groupby('customer_id').agg({
        'revenue': 'sum',
        'conversion': 'sum',
        'engagement_date': 'count'
    }).reset_index()
    clv_data.columns = ['customer_id', 'total_revenue', 'total_conversions', 'total_engagements']
    clv_data = clv_data.merge(filtered_customers[['customer_id', 'loyalty_status', 'income_level']], on='customer_id')
    
    fig_clv = px.scatter(
        clv_data,
        x='total_engagements',
        y='total_revenue',
        color='loyalty_status',
        size='total_conversions',
        hover_data=['customer_id', 'income_level'],
        title='Customer Lifetime Value: Engagements vs Revenue',
        labels={'total_engagements': 'Total Engagements', 'total_revenue': 'Total Revenue ($)'}
    )
    st.plotly_chart(fig_clv, use_container_width=True)
    
    # Loyalty Status Breakdown
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Loyalty Status Distribution")
        loyalty_dist = filtered_customers['loyalty_status'].value_counts().reset_index()
        loyalty_dist.columns = ['loyalty_status', 'count']
        fig_loyalty = px.pie(
            loyalty_dist,
            values='count',
            names='loyalty_status',
            title='Customer Loyalty Distribution',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_loyalty, use_container_width=True)
    
    with col2:
        st.subheader("Income Level vs Loyalty")
        income_loyalty = filtered_customers.groupby(['income_level', 'loyalty_status']).size().reset_index(name='count')
        fig_income_loyalty = px.bar(
            income_loyalty,
            x='income_level',
            y='count',
            color='loyalty_status',
            title='Income Level vs Loyalty Status',
            barmode='group'
        )
        st.plotly_chart(fig_income_loyalty, use_container_width=True)

# ==================== PRODUCT PERFORMANCE PAGE ====================
elif page == "Product Performance":
    st.header("📦 Product Performance Analysis")
    
    # Product filters
    col1, col2 = st.columns(2)
    with col1:
        category_filter = st.multiselect(
            "Product Category",
            options=products['category'].unique(),
            default=products['category'].unique()
        )
    with col2:
        price_range = st.slider(
            "Price Range ($)",
            min_value=float(products['price'].min()),
            max_value=float(products['price'].max()),
            value=(float(products['price'].min()), float(products['price'].max()))
        )
    
    # Filter products
    filtered_products = products[
        (products['category'].isin(category_filter)) &
        (products['price'] >= price_range[0]) &
        (products['price'] <= price_range[1])
    ]
    
    filtered_df_prod = df_merged[df_merged['product_id'].isin(filtered_products['product_id'])]
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Products", f"{filtered_products['product_id'].nunique()}")
    with col2:
        total_prod_revenue = filtered_df_prod['revenue'].sum()
        st.metric("Total Revenue", f"${total_prod_revenue:,.2f}")
    with col3:
        avg_price = filtered_products['price'].mean()
        st.metric("Avg Product Price", f"${avg_price:.2f}")
    with col4:
        total_conversions = filtered_df_prod['conversion'].sum()
        st.metric("Total Conversions", f"{int(total_conversions):,}")
    
    st.markdown("---")
    
    # Product performance table
    st.subheader("Top Performing Products")
    product_performance = filtered_df_prod.groupby('product_id').agg({
        'revenue': 'sum',
        'conversion': 'sum',
        'customer_id': 'count'
    }).reset_index()
    product_performance.columns = ['product_id', 'total_revenue', 'conversions', 'engagements']
    product_performance = product_performance.merge(filtered_products[['product_id', 'product_name', 'category', 'price']], on='product_id')
    product_performance['conversion_rate'] = (product_performance['conversions'] / product_performance['engagements'] * 100).round(2)
    product_performance = product_performance.sort_values('total_revenue', ascending=False)
    
    st.dataframe(
        product_performance[['product_name', 'category', 'price', 'total_revenue', 'conversions', 'conversion_rate']].head(10),
        use_container_width=True
    )
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Revenue by Category")
        category_revenue = filtered_df_prod.groupby('category')['revenue'].sum().reset_index().sort_values('revenue', ascending=False)
        fig_cat_rev = px.bar(
            category_revenue,
            x='category',
            y='revenue',
            title='Total Revenue by Product Category',
            labels={'category': 'Category', 'revenue': 'Revenue ($)'},
            color='revenue',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_cat_rev, use_container_width=True)
    
    with col2:
        st.subheader("Conversions by Category")
        category_conv = filtered_df_prod.groupby('category')['conversion'].sum().reset_index().sort_values('conversion', ascending=False)
        fig_cat_conv = px.bar(
            category_conv,
            x='category',
            y='conversion',
            title='Total Conversions by Category',
            labels={'category': 'Category', 'conversion': 'Conversions'},
            color='conversion',
            color_continuous_scale='Oranges'
        )
        st.plotly_chart(fig_cat_conv, use_container_width=True)
    
    # Price vs Revenue Analysis
    st.subheader("Price vs Revenue Analysis")
    price_revenue = product_performance.copy()
    fig_price = px.scatter(
        price_revenue,
        x='price',
        y='total_revenue',
        size='conversions',
        color='category',
        hover_data=['product_name'],
        title='Product Price vs Total Revenue',
        labels={'price': 'Product Price ($)', 'total_revenue': 'Total Revenue ($)'}
    )
    st.plotly_chart(fig_price, use_container_width=True)
    
    # Category performance comparison
    st.subheader("Category Performance Comparison")
    category_metrics = filtered_df_prod.groupby('category').agg({
        'revenue': 'sum',
        'conversion': 'sum',
        'customer_id': 'count'
    }).reset_index()
    category_metrics.columns = ['category', 'revenue', 'conversions', 'engagements']
    category_metrics['conversion_rate'] = (category_metrics['conversions'] / category_metrics['engagements'] * 100).round(2)
    
    fig_cat_comp = go.Figure()
    fig_cat_comp.add_trace(go.Bar(
        x=category_metrics['category'],
        y=category_metrics['revenue'],
        name='Revenue',
        yaxis='y',
        marker_color='#1f77b4'
    ))
    fig_cat_comp.add_trace(go.Scatter(
        x=category_metrics['category'],
        y=category_metrics['conversion_rate'],
        name='Conversion Rate (%)',
        yaxis='y2',
        mode='lines+markers',
        marker_color='#ff7f0e',
        line=dict(width=3)
    ))
    fig_cat_comp.update_layout(
        title='Category Revenue and Conversion Rate',
        xaxis=dict(title='Category'),
        yaxis=dict(title='Revenue ($)', side='left'),
        yaxis2=dict(title='Conversion Rate (%)', overlaying='y', side='right'),
        legend=dict(x=0.01, y=0.99)
    )
    st.plotly_chart(fig_cat_comp, use_container_width=True)

# ==================== ENGAGEMENT INSIGHTS PAGE ====================
elif page == "Engagement Insights":
    st.header("💬 Engagement Insights")
    
    # Date range filter
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=engagement['engagement_date'].min()
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=engagement['engagement_date'].max()
        )
    
    # Channel filter
    channel_filter = st.multiselect(
        "Select Channels",
        options=engagement['channel'].unique(),
        default=engagement['channel'].unique()
    )
    
    # Filter engagement data
    filtered_engagement = df_merged[
        (df_merged['engagement_date'] >= pd.to_datetime(start_date)) &
        (df_merged['engagement_date'] <= pd.to_datetime(end_date)) &
        (df_merged['channel'].isin(channel_filter))
    ]
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_eng = len(filtered_engagement)
        st.metric("Total Engagements", f"{total_eng:,}")
    with col2:
        total_conv = filtered_engagement['conversion'].sum()
        st.metric("Total Conversions", f"{int(total_conv):,}")
    with col3:
        conv_rate = (total_conv / total_eng * 100) if total_eng > 0 else 0
        st.metric("Conversion Rate", f"{conv_rate:.2f}%")
    with col4:
        eng_revenue = filtered_engagement['revenue'].sum()
        st.metric("Revenue", f"${eng_revenue:,.2f}")
    
    st.markdown("---")
    
    # Engagement over time
    st.subheader("Engagement Trends Over Time")
    daily_engagement = filtered_engagement.groupby(filtered_engagement['engagement_date'].dt.date).agg({
        'customer_id': 'count',
        'conversion': 'sum',
        'revenue': 'sum'
    }).reset_index()
    daily_engagement.columns = ['date', 'engagements', 'conversions', 'revenue']
    
    fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
    fig_trend.add_trace(
        go.Scatter(x=daily_engagement['date'], y=daily_engagement['engagements'], name='Engagements', line=dict(color='#1f77b4')),
        secondary_y=False
    )
    fig_trend.add_trace(
        go.Scatter(x=daily_engagement['date'], y=daily_engagement['conversions'], name='Conversions', line=dict(color='#ff7f0e')),
        secondary_y=True
    )
    fig_trend.update_layout(title='Daily Engagements and Conversions')
    fig_trend.update_xaxes(title_text='Date')
    fig_trend.update_yaxes(title_text='Engagements', secondary_y=False)
    fig_trend.update_yaxes(title_text='Conversions', secondary_y=True)
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Channel analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Engagement by Channel")
        channel_eng = filtered_engagement.groupby('channel').size().reset_index(name='count')
        fig_channel_eng = px.pie(
            channel_eng,
            values='count',
            names='channel',
            title='Engagement Distribution by Channel',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_channel_eng, use_container_width=True)
    
    with col2:
        st.subheader("Conversion Rate by Channel")
        channel_conv = filtered_engagement.groupby('channel').agg({
            'conversion': 'sum',
            'customer_id': 'count'
        }).reset_index()
        channel_conv['conversion_rate'] = (channel_conv['conversion'] / channel_conv['customer_id'] * 100).round(2)
        fig_channel_conv = px.bar(
            channel_conv,
            x='channel',
            y='conversion_rate',
            title='Conversion Rate by Channel (%)',
            labels={'channel': 'Channel', 'conversion_rate': 'Conversion Rate (%)'},
            color='conversion_rate',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig_channel_conv, use_container_width=True)
    
    # Hourly engagement pattern (if time data available)
    if 'engagement_date' in filtered_engagement.columns:
        st.subheader("Engagement Patterns")
        filtered_engagement['hour'] = filtered_engagement['engagement_date'].dt.hour
        filtered_engagement['day_of_week'] = filtered_engagement['engagement_date'].dt.day_name()
        
        col1, col2 = st.columns(2)
        
        with col1:
            hourly_eng = filtered_engagement.groupby('hour').size().reset_index(name='count')
            fig_hourly = px.line(
                hourly_eng,
                x='hour',
                y='count',
                title='Engagement by Hour of Day',
                labels={'hour': 'Hour', 'count': 'Engagements'},
                markers=True
            )
            st.plotly_chart(fig_hourly, use_container_width=True)
        
        with col2:
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily_eng = filtered_engagement.groupby('day_of_week').size().reset_index(name='count')
            daily_eng['day_of_week'] = pd.Categorical(daily_eng['day_of_week'], categories=day_order, ordered=True)
            daily_eng = daily_eng.sort_values('day_of_week')
            fig_daily = px.bar(
                daily_eng,
                x='day_of_week',
                y='count',
                title='Engagement by Day of Week',
                labels={'day_of_week': 'Day', 'count': 'Engagements'},
                color='count',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_daily, use_container_width=True)
    
    # Channel performance detailed table
    st.subheader("Detailed Channel Performance")
    channel_detail = filtered_engagement.groupby('channel').agg({
        'customer_id': 'count',
        'conversion': 'sum',
        'revenue': 'sum'
    }).reset_index()
    channel_detail.columns = ['channel', 'engagements', 'conversions', 'revenue']
    channel_detail['conversion_rate'] = (channel_detail['conversions'] / channel_detail['engagements'] * 100).round(2)
    channel_detail['revenue_per_engagement'] = (channel_detail['revenue'] / channel_detail['engagements']).round(2)
    channel_detail = channel_detail.sort_values('revenue', ascending=False)
    
    st.dataframe(channel_detail, use_container_width=True)

# ==================== GEOGRAPHIC ANALYSIS PAGE ====================
elif page == "Geographic Analysis":
    st.header("🌍 Geographic Analysis")
    
    # Region filter
    region_filter = st.multiselect(
        "Select Regions",
        options=geography['region'].unique(),
        default=geography['region'].unique()
    )
    
    # Filter data
    filtered_geo = df_merged[df_merged['region'].isin(region_filter)]
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        unique_regions = filtered_geo['region'].nunique()
        st.metric("Regions", f"{unique_regions}")
    with col2:
        unique_countries = filtered_geo['country'].nunique()
        st.metric("Countries", f"{unique_countries}")
    with col3:
        geo_revenue = filtered_geo['revenue'].sum()
        st.metric("Total Revenue", f"${geo_revenue:,.2f}")
    with col4:
        geo_customers = filtered_geo['customer_id'].nunique()
        st.metric("Customers", f"{geo_customers:,}")
    
    st.markdown("---")
    
    # Regional performance
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Revenue by Region")
        region_revenue = filtered_geo.groupby('region')['revenue'].sum().reset_index().sort_values('revenue', ascending=False)
        fig_region_rev = px.bar(
            region_revenue,
            x='region',
            y='revenue',
            title='Total Revenue by Region',
            labels={'region': 'Region', 'revenue': 'Revenue ($)'},
            color='revenue',
            color_continuous_scale='Teal'
        )
        st.plotly_chart(fig_region_rev, use_container_width=True)
    
    with col2:
        st.subheader("Customer Distribution by Region")
        region_customers = filtered_geo.groupby('region')['customer_id'].nunique().reset_index()
        region_customers.columns = ['region', 'customers']
        fig_region_cust = px.pie(
            region_customers,
            values='customers',
            names='region',
            title='Customer Distribution by Region',
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        st.plotly_chart(fig_region_cust, use_container_width=True)
    
    # Country analysis
    st.subheader("Top Countries by Revenue")
    country_revenue = filtered_geo.groupby('country').agg({
        'revenue': 'sum',
        'customer_id': 'nunique',
        'conversion': 'sum'
    }).reset_index()
    country_revenue.columns = ['country', 'revenue', 'customers', 'conversions']
    country_revenue = country_revenue.sort_values('revenue', ascending=False).head(15)
    
    fig_country = px.bar(
        country_revenue,
        x='country',
        y='revenue',
        title='Top 15 Countries by Revenue',
        labels={'country': 'Country', 'revenue': 'Revenue ($)'},
        color='revenue',
        color_continuous_scale='Sunset',
        hover_data=['customers', 'conversions']
    )
    fig_country.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_country, use_container_width=True)
    
    # Regional performance metrics
    st.subheader("Regional Performance Metrics")
    regional_metrics = filtered_geo.groupby('region').agg({
        'revenue': 'sum',
        'customer_id': 'nunique',
        'conversion': 'sum',
        'product_id': 'count'
    }).reset_index()
    regional_metrics.columns = ['region', 'revenue', 'customers', 'conversions', 'engagements']
    regional_metrics['avg_revenue_per_customer'] = (regional_metrics['revenue'] / regional_metrics['customers']).round(2)
    regional_metrics['conversion_rate'] = (regional_metrics['conversions'] / regional_metrics['engagements'] * 100).round(2)
    regional_metrics = regional_metrics.sort_values('revenue', ascending=False)
    
    st.dataframe(regional_metrics, use_container_width=True)
    
    # Country vs Region heatmap
    st.subheader("Revenue Heatmap: Region vs Country (Top 10)")
    top_countries = filtered_geo.groupby('country')['revenue'].sum().nlargest(10).index
    heatmap_data = filtered_geo[filtered_geo['country'].isin(top_countries)].groupby(['region', 'country'])['revenue'].sum().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='region', columns='country', values='revenue').fillna(0)
    
    fig_heatmap = px.imshow(
        heatmap_pivot,
        title='Revenue Heatmap: Region vs Top 10 Countries',
        labels=dict(x='Country', y='Region', color='Revenue ($)'),
        color_continuous_scale='YlOrRd',
        aspect='auto'
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

# ==================== CAMPAIGN ROI PAGE ====================
elif page == "Campaign ROI":
    st.header("💰 Campaign ROI Analysis")
    
    st.info("📊 This page analyzes the Return on Investment (ROI) for marketing campaigns across different channels and segments.")
    
    # Calculate ROI metrics
    # Assuming marketing spend is proportional to engagements (you can adjust this logic)
    avg_cost_per_engagement = 5  # Example: $5 per engagement
    
    df_merged['marketing_cost'] = avg_cost_per_engagement
    
    # Overall ROI
    total_revenue = df_merged['revenue'].sum()
    total_cost = len(df_merged) * avg_cost_per_engagement
    overall_roi = ((total_revenue - total_cost) / total_cost * 100) if total_cost > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Revenue", f"${total_revenue:,.2f}")
    with col2:
        st.metric("Total Marketing Cost", f"${total_cost:,.2f}")
    with col3:
        st.metric("Net Profit", f"${total_revenue - total_cost:,.2f}")
    with col4:
        st.metric("Overall ROI", f"{overall_roi:.2f}%")
    
    st.markdown("---")
    
    # ROI by Channel
    st.subheader("ROI by Marketing Channel")
    channel_roi = df_merged.groupby('channel').agg({
        'revenue': 'sum',
        'customer_id': 'count'
    }).reset_index()
    channel_roi.columns = ['channel', 'revenue', 'engagements']
    channel_roi['cost'] = channel_roi['engagements'] * avg_cost_per_engagement
    channel_roi['profit'] = channel_roi['revenue'] - channel_roi['cost']
    channel_roi['roi'] = ((channel_roi['profit'] / channel_roi['cost']) * 100).round(2)
    channel_roi = channel_roi.sort_values('roi', ascending=False)
    
    fig_channel_roi = px.bar(
        channel_roi,
        x='channel',
        y='roi',
        title='ROI by Channel (%)',
        labels={'channel': 'Channel', 'roi': 'ROI (%)'},
        color='roi',
        color_continuous_scale='RdYlGn',
        text='roi'
    )
    fig_channel_roi.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    st.plotly_chart(fig_channel_roi, use_container_width=True)
    
    # Detailed channel ROI table
    st.dataframe(channel_roi[['channel', 'revenue', 'cost', 'profit', 'roi']].style.format({
        'revenue': '${:,.2f}',
        'cost': '${:,.2f}',
        'profit': '${:,.2f}',
        'roi': '{:.2f}%'
    }), use_container_width=True)
    
    # ROI by Product Category
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("ROI by Product Category")
        category_roi = df_merged.groupby('category').agg({
            'revenue': 'sum',
            'customer_id': 'count'
        }).reset_index()
        category_roi.columns = ['category', 'revenue', 'engagements']
        category_roi['cost'] = category_roi['engagements'] * avg_cost_per_engagement
        category_roi['roi'] = (((category_roi['revenue'] - category_roi['cost']) / category_roi['cost']) * 100).round(2)
        
        fig_cat_roi = px.bar(
            category_roi.sort_values('roi', ascending=True),
            y='category',
            x='roi',
            orientation='h',
            title='ROI by Product Category',
            labels={'category': 'Category', 'roi': 'ROI (%)'},
            color='roi',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig_cat_roi, use_container_width=True)
    
    with col2:
        st.subheader("ROI by Customer Segment")
        segment_roi = df_merged.groupby('income_level').agg({
            'revenue': 'sum',
            'customer_id': 'count'
        }).reset_index()
        segment_roi.columns = ['income_level', 'revenue', 'engagements']
        segment_roi['cost'] = segment_roi['engagements'] * avg_cost_per_engagement
        segment_roi['roi'] = (((segment_roi['revenue'] - segment_roi['cost']) / segment_roi['cost']) * 100).round(2)
        
        fig_seg_roi = px.bar(
            segment_roi.sort_values('roi', ascending=True),
            y='income_level',
            x='roi',
            orientation='h',
            title='ROI by Income Level',
            labels={'income_level': 'Income Level', 'roi': 'ROI (%)'},
            color='roi',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_seg_roi, use_container_width=True)
    
    # ROI Trend Over Time
    st.subheader("ROI Trend Over Time")
    monthly_roi = df_merged.groupby(df_merged['engagement_date'].dt.to_period('M')).agg({
        'revenue': 'sum',
        'customer_id': 'count'
    }).reset_index()
    monthly_roi.columns = ['month', 'revenue', 'engagements']
    monthly_roi['month'] = monthly_roi['month'].astype(str)
    monthly_roi['cost'] = monthly_roi['engagements'] * avg_cost_per_engagement
    monthly_roi['roi'] = (((monthly_roi['revenue'] - monthly_roi['cost']) / monthly_roi['cost']) * 100).round(2)
    
    fig_roi_trend = go.Figure()
    fig_roi_trend.add_trace(go.Scatter(
        x=monthly_roi['month'],
        y=monthly_roi['roi'],
        mode='lines+markers',
        name='ROI',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    fig_roi_trend.update_layout(
        title='Monthly ROI Trend',
        xaxis_title='Month',
        yaxis_title='ROI (%)',
        hovermode='x unified'
    )
    st.plotly_chart(fig_roi_trend, use_container_width=True)
    
    # Cost-Benefit Analysis
    st.subheader("Cost-Benefit Analysis by Region")
    region_roi = df_merged.groupby('region').agg({
        'revenue': 'sum',
        'customer_id': 'count',
        'conversion': 'sum'
    }).reset_index()
    region_roi.columns = ['region', 'revenue', 'engagements', 'conversions']
    region_roi['cost'] = region_roi['engagements'] * avg_cost_per_engagement
    region_roi['profit'] = region_roi['revenue'] - region_roi['cost']
    region_roi['roi'] = ((region_roi['profit'] / region_roi['cost']) * 100).round(2)
    region_roi['cost_per_conversion'] = (region_roi['cost'] / region_roi['conversions']).round(2)
    
    fig_region_roi = px.scatter(
        region_roi,
        x='cost',
        y='revenue',
        size='conversions',
        color='roi',
        hover_data=['region', 'profit'],
        title='Cost vs Revenue by Region (bubble size = conversions)',
        labels={'cost': 'Marketing Cost ($)', 'revenue': 'Revenue ($)'},
        color_continuous_scale='RdYlGn'
    )
    # Add diagonal line for break-even
    max_val = max(region_roi['cost'].max(), region_roi['revenue'].max())
    fig_region_roi.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        name='Break-even',
        line=dict(dash='dash', color='red')
    ))
    st.plotly_chart(fig_region_roi, use_container_width=True)
    
    # Key Insights
    st.subheader("💡 Key Insights")
    best_channel = channel_roi.loc[channel_roi['roi'].idxmax(), 'channel']
    best_channel_roi = channel_roi.loc[channel_roi['roi'].idxmax(), 'roi']
    
    worst_channel = channel_roi.loc[channel_roi['roi'].idxmin(), 'channel']
    worst_channel_roi = channel_roi.loc[channel_roi['roi'].idxmin(), 'roi']
    
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"🏆 **Best Performing Channel:** {best_channel} with {best_channel_roi:.2f}% ROI")
    with col2:
        st.warning(f"⚠️ **Lowest Performing Channel:** {worst_channel} with {worst_channel_roi:.2f}% ROI")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #7f8c8d;'>
        <p>Marketing Campaign Analytics Dashboard | Built with Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
