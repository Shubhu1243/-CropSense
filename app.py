import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# MUST be the first Streamlit command
st.set_page_config(
    page_title="Crop Yield Predictor",
    page_icon="🌾",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        transform: translateY(-2px);
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)


# Load models
@st.cache_resource
def load_models():
    try:
        dtr = pickle.load(open('models/dtr.pkl', 'rb'))
        preprocessor = pickle.load(open('models/preprocessor.pkl', 'rb'))
        return dtr, preprocessor, None
    except FileNotFoundError as e:
        return None, None, f"Model file not found: {str(e)}"
    except Exception as e:
        return None, None, f"Error loading models: {str(e)}"


# Load dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('yield_df.csv')
        return df, None
    except FileNotFoundError:
        return None, "Dataset 'yield_df.csv' not found"
    except Exception as e:
        return None, f"Error loading data: {str(e)}"


# Initialize
dtr, preprocessor, model_error = load_models()
df, data_error = load_data()

# Sidebar Navigation
st.sidebar.title("🌾 Crop Yield Predictor")
st.sidebar.markdown("---")

if model_error:
    st.sidebar.error("⚠️ Model Error")
else:
    st.sidebar.success("✅ Models Loaded")

if data_error:
    st.sidebar.warning("⚠️ Dataset Missing")
else:
    st.sidebar.success(f"✅ Dataset Loaded ({len(df)} rows)")

page = st.sidebar.radio(
    "📍 Navigation",
    ["🏠 Home", "🔮 Prediction", "📊 Data Explorer", "📈 Analytics", "ℹ️ About"]
)

st.sidebar.markdown("---")
st.sidebar.info("**Model:** Decision Tree Regressor")

# ============= HOME PAGE =============
if page == "🏠 Home":
    st.title("🌾 Crop Yield Prediction System")
    st.markdown("### AI-Powered Agricultural Intelligence Platform")

    if model_error:
        st.error(f"❌ {model_error}")
        st.info("Please ensure 'models/dtr.pkl' and 'models/preprocessor.pkl' exist.")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h3 style='text-align: center;'>🤖</h3>
            <h4 style='text-align: center;'>ML Model</h4>
            <p style='text-align: center;'>Decision Tree</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='metric-card'>
            <h3 style='text-align: center;'>⚡</h3>
            <h4 style='text-align: center;'>Fast</h4>
            <p style='text-align: center;'>Instant Results</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='metric-card'>
            <h3 style='text-align: center;'>🎯</h3>
            <h4 style='text-align: center;'>Accurate</h4>
            <p style='text-align: center;'>Reliable Predictions</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class='metric-card'>
            <h3 style='text-align: center;'>📊</h3>
            <h4 style='text-align: center;'>Analytics</h4>
            <p style='text-align: center;'>Data Insights</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 Features")
        st.markdown("""
        - **Real-time Predictions** - Get instant yield forecasts
        - **Multiple Crops** - Supports various crop types
        - **Global Coverage** - Works for different regions
        - **Interactive Dashboard** - Explore your data visually
        - **Historical Analysis** - Analyze trends over time
        """)

    with col2:
        st.markdown("### 🚀 How to Use")
        st.markdown("""
        1. Go to **🔮 Prediction** page
        2. Select Area (Country)
        3. Select Crop Type (Item)
        4. Enter Year and environmental parameters
        5. Click **Predict** to get results
        """)

        if df is not None:
            st.markdown("### 📊 Dataset Info")
            st.info(f"**Total Records:** {len(df):,}")
            st.info(f"**Countries:** {df['Area'].nunique()}")
            st.info(f"**Crop Types:** {df['Item'].nunique()}")

# ============= PREDICTION PAGE =============
elif page == "🔮 Prediction":
    st.title("🔮 Crop Yield Prediction")
    st.markdown("### Enter parameters to get yield predictions")

    if model_error:
        st.error(f"❌ {model_error}")
        st.stop()

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("### 📝 Input Parameters")

        # Get unique values from dataset
        if df is not None:
            areas = sorted(df['Area'].unique().tolist())
            items = sorted(df['Item'].unique().tolist())
        else:
            areas = ['India', 'USA', 'China', 'Albania', 'Zimbabwe']
            items = ['Wheat', 'Rice, paddy', 'Maize', 'Soybeans', 'Potatoes']

        # Two column layout for inputs
        input_col1, input_col2 = st.columns(2)

        with input_col1:
            area = st.selectbox(
                "🌍 Area (Country/Region)",
                options=areas,
                index=0
            )

            item = st.selectbox(
                "🌾 Item (Crop Type)",
                options=items,
                index=0
            )

            year = st.number_input(
                "📅 Year",
                min_value=1990,
                max_value=2030,
                value=2024,
                step=1
            )

        with input_col2:
            avg_rainfall = st.number_input(
                "🌧️ Average Rainfall (mm/year)",
                min_value=0.0,
                max_value=5000.0,
                value=1000.0,
                step=10.0
            )

            pesticides = st.number_input(
                "🧪 Pesticides (tonnes)",
                min_value=0.0,
                max_value=10000.0,
                value=121.0,
                step=1.0
            )

            avg_temp = st.number_input(
                "🌡️ Average Temperature (°C)",
                min_value=-20.0,
                max_value=50.0,
                value=20.0,
                step=0.5
            )

        st.markdown("---")
        predict_button = st.button("🎯 Predict Yield", type="primary")

    with col2:
        st.markdown("### 📊 Prediction Results")

        if predict_button:
            try:
                # Prepare features in the exact order expected by the model
                features = np.array([[year, avg_rainfall, pesticides, avg_temp, area, item]], dtype=object)

                # Transform and predict
                transformed_features = preprocessor.transform(features)
                prediction = dtr.predict(transformed_features)[0]

                # Display prediction
                st.markdown(f"""
                <div class='prediction-box'>
                    <h3 style='margin: 0; color: white;'>Predicted Yield</h3>
                    <h1 style='margin: 15px 0; font-size: 3.5em;'>{prediction:,.2f}</h1>
                    <p style='margin: 0; font-size: 1.1em; opacity: 0.9;'>hg/ha (hectograms/hectare)</p>
                </div>
                """, unsafe_allow_html=True)

                st.success("✅ Prediction completed successfully!")

                # Show input summary
                st.markdown("---")
                st.markdown("### 📋 Input Summary")
                summary_df = pd.DataFrame({
                    'Parameter': ['Area', 'Crop Type', 'Year', 'Rainfall', 'Pesticides', 'Temperature'],
                    'Value': [area, item, year, f"{avg_rainfall} mm", f"{pesticides} tonnes", f"{avg_temp} °C"]
                })
                st.dataframe(summary_df, hide_index=True, use_container_width=True)

                # Comparison with historical data
                if df is not None:
                    st.markdown("---")
                    st.markdown("### 📊 Historical Comparison")

                    # Filter data for same area and item
                    filtered_df = df[(df['Area'] == area) & (df['Item'] == item)]

                    if len(filtered_df) > 0:
                        avg_yield = filtered_df['hg/ha_yield'].mean()
                        max_yield = filtered_df['hg/ha_yield'].max()
                        min_yield = filtered_df['hg/ha_yield'].min()

                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Average", f"{avg_yield:,.0f}")
                        col_b.metric("Maximum", f"{max_yield:,.0f}")
                        col_c.metric("Minimum", f"{min_yield:,.0f}")

                        # Performance indicator
                        if prediction > avg_yield:
                            st.success(
                                f"🎉 Predicted yield is {((prediction / avg_yield - 1) * 100):.1f}% above historical average!")
                        else:
                            st.info(
                                f"📊 Predicted yield is {((1 - prediction / avg_yield) * 100):.1f}% below historical average")
                    else:
                        st.info("No historical data available for this combination")

            except Exception as e:
                st.error(f"❌ Prediction Error: {str(e)}")
                st.warning("💡 **Tip:** Ensure Area and Item values match those used during training")
                st.code(f"Error details: {type(e).__name__}: {str(e)}")
        else:
            st.info("👈 Fill in the parameters and click **Predict Yield**")

            if df is not None:
                st.markdown("---")
                st.markdown("### 💡 Quick Stats")
                st.metric("Total Records", f"{len(df):,}")
                st.metric("Countries Available", df['Area'].nunique())
                st.metric("Crop Types Available", df['Item'].nunique())

# ============= DATA EXPLORER PAGE =============
elif page == "📊 Data Explorer":
    st.title("📊 Data Explorer")
    st.markdown("### Explore the dataset interactively")

    if df is None:
        st.error(f"❌ {data_error}")
        st.info("Please ensure 'yield_df.csv' exists in your project folder")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["📋 Dataset", "🔍 Filters", "📈 Visualizations"])

    with tab1:
        st.markdown("### Dataset Overview")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", f"{len(df):,}")
        col2.metric("Columns", len(df.columns))
        col3.metric("Countries", df['Area'].nunique())
        col4.metric("Crops", df['Item'].nunique())

        st.markdown("---")
        st.markdown("### Data Preview")
        st.dataframe(df.head(100), use_container_width=True, height=400)

        st.markdown("---")
        st.markdown("### Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)

    with tab2:
        st.markdown("### Filter Data")

        col1, col2 = st.columns(2)

        with col1:
            selected_areas = st.multiselect(
                "Select Areas",
                options=df['Area'].unique().tolist(),
                default=df['Area'].unique().tolist()[:5]
            )

        with col2:
            selected_items = st.multiselect(
                "Select Crop Types",
                options=df['Item'].unique().tolist(),
                default=df['Item'].unique().tolist()[:5]
            )

        year_range = st.slider(
            "Select Year Range",
            min_value=int(df['Year'].min()),
            max_value=int(df['Year'].max()),
            value=(int(df['Year'].min()), int(df['Year'].max()))
        )

        # Apply filters
        filtered_df = df[
            (df['Area'].isin(selected_areas)) &
            (df['Item'].isin(selected_items)) &
            (df['Year'] >= year_range[0]) &
            (df['Year'] <= year_range[1])
            ]

        st.markdown(f"### Filtered Results: {len(filtered_df)} records")
        st.dataframe(filtered_df, use_container_width=True, height=400)

        # Download button
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Filtered Data",
            csv,
            "filtered_crop_data.csv",
            "text/csv",
            key='download-csv'
        )

    with tab3:
        st.markdown("### Data Visualizations")

        # Yield by Country
        st.markdown("#### Average Yield by Country (Top 10)")
        top_countries = df.groupby('Area')['hg/ha_yield'].mean().sort_values(ascending=False).head(10)
        fig1 = px.bar(
            x=top_countries.values,
            y=top_countries.index,
            orientation='h',
            labels={'x': 'Average Yield (hg/ha)', 'y': 'Country'},
            color=top_countries.values,
            color_continuous_scale='Viridis'
        )
        fig1.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)

        # Yield by Crop
        st.markdown("#### Average Yield by Crop Type (Top 10)")
        top_crops = df.groupby('Item')['hg/ha_yield'].mean().sort_values(ascending=False).head(10)
        fig2 = px.bar(
            x=top_crops.index,
            y=top_crops.values,
            labels={'x': 'Crop Type', 'y': 'Average Yield (hg/ha)'},
            color=top_crops.values,
            color_continuous_scale='Blues'
        )
        fig2.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

        # Yield over time
        st.markdown("#### Yield Trend Over Years")
        yearly_avg = df.groupby('Year')['hg/ha_yield'].mean().reset_index()
        fig3 = px.line(
            yearly_avg,
            x='Year',
            y='hg/ha_yield',
            labels={'hg/ha_yield': 'Average Yield (hg/ha)'},
            markers=True
        )
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)

# ============= ANALYTICS PAGE =============
elif page == "📈 Analytics":
    st.title("📈 Advanced Analytics")
    st.markdown("### Deep insights from the data")

    if df is None:
        st.error(f"❌ {data_error}")
        st.stop()

    tab1, tab2 = st.tabs(["📊 Key Metrics", "🔗 Correlations"])

    with tab1:
        st.markdown("### Key Performance Indicators")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Average Yield",
            f"{df['hg/ha_yield'].mean():,.0f}",
            "hg/ha"
        )
        col2.metric(
            "Max Yield",
            f"{df['hg/ha_yield'].max():,.0f}",
            "hg/ha"
        )
        col3.metric(
            "Avg Rainfall",
            f"{df['average_rain_fall_mm_per_year'].mean():,.0f}",
            "mm/year"
        )
        col4.metric(
            "Avg Temperature",
            f"{df['avg_temp'].mean():.1f}",
            "°C"
        )

        st.markdown("---")

        # Distribution plots
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Yield Distribution")
            fig = px.histogram(
                df,
                x='hg/ha_yield',
                nbins=50,
                labels={'hg/ha_yield': 'Yield (hg/ha)'},
                color_discrete_sequence=['#667eea']
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Rainfall Distribution")
            fig = px.histogram(
                df,
                x='average_rain_fall_mm_per_year',
                nbins=50,
                labels={'average_rain_fall_mm_per_year': 'Rainfall (mm/year)'},
                color_discrete_sequence=['#764ba2']
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("### Correlation Analysis")

        # Correlation heatmap
        numeric_cols = ['Year', 'hg/ha_yield', 'average_rain_fall_mm_per_year',
                        'pesticides_tonnes', 'avg_temp']
        corr_matrix = df[numeric_cols].corr()

        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            aspect='auto',
            labels=dict(color="Correlation")
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Scatter plots
        st.markdown("---")
        st.markdown("### Relationship Analysis")

        col1, col2 = st.columns(2)
        with col1:
            x_var = st.selectbox("X-axis", numeric_cols, index=2)
        with col2:
            y_var = st.selectbox("Y-axis", numeric_cols, index=1)

        fig = px.scatter(
            df.sample(min(1000, len(df))),
            x=x_var,
            y=y_var,
            color='avg_temp',
            color_continuous_scale='Viridis',
            opacity=0.6
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

# ============= ABOUT PAGE =============
else:
    st.title("ℹ️ About This Application")
    st.markdown("### Crop Yield Prediction System")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### 🎯 Project Overview
        This application uses Machine Learning to predict crop yields based on:
        - Location (Country/Region)
        - Crop Type
        - Year
        - Environmental factors (Rainfall, Temperature)
        - Pesticide usage

        #### 🔧 Technology Stack
        - **Framework:** Streamlit
        - **ML Model:** Decision Tree Regressor
        - **Libraries:** Pandas, NumPy, Scikit-learn, Plotly
        - **Data Processing:** Custom preprocessing pipeline

        #### 📊 Dataset
        - **28,242 records** across multiple countries and years
        - **7 features** including environmental and agricultural data
        - **Multiple crop types** from around the world
        """)

    with col2:
        st.markdown("""
        #### 🚀 Features
        - **Real-time Predictions** with instant results
        - **Interactive Data Explorer** to analyze patterns
        - **Visual Analytics** with charts and graphs
        - **Historical Comparisons** to benchmark predictions
        - **Multi-country Support** for global agriculture

        #### 📖 How to Use
        1. Navigate to **🔮 Prediction** page
        2. Select your area and crop type
        3. Enter environmental parameters
        4. Click **Predict** to see results
        5. Explore **📊 Data Explorer** for insights

        #### 💡 Tips for Best Results
        - Use values within historical ranges
        - Check similar historical records for reference
        - Compare predictions with historical averages
        """)

    st.markdown("---")
    st.success("🌾 Built with ❤️ for Agricultural Innovation")

    if df is not None:
        st.markdown("### 📊 Dataset Statistics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", f"{len(df):,}")
        col2.metric("Date Range", f"{df['Year'].min()} - {df['Year'].max()}")
        col3.metric("Countries", df['Area'].nunique())
        col4.metric("Crop Types", df['Item'].nunique())

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 🌾 Crop Yield Predictor")
st.sidebar.caption("v1.0 | Built with Streamlit")
st.sidebar.caption("© 2024 Agricultural Intelligence")