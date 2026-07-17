import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# 1. Page Configuration
st.set_page_config(
    page_title="Universal CSV Analytics Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Header
st.title("📊 Universal CSV Analytics Engine")
st.markdown("Upload any CSV dataset to instantly generate dynamic summaries, high-density matrix visualizations, and anomaly reports.")
st.markdown("---")

# 3. File Uploader Component
uploaded_file = st.sidebar.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load data
    @st.cache_data
    def load_uploaded_data(file):
        df = pd.read_csv(file)
        return df

    df = load_uploaded_data(uploaded_file)
    
    # Identify Column Types Automatically
    all_cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=[object, 'category']).columns.tolist()

    # If pandas loaded everything as object, fallback to all columns
    if not categorical_cols:
        categorical_cols = all_cols

    # 4. Dynamic Column Mapping Sidebar
    st.sidebar.markdown("### ⚙️ Mapping Configurations")
    st.sidebar.info("Select which columns represent your primary metrics and categories for the advanced charts.")
    
    # Fallbacks to prevent crashing if lists are empty
    target_numeric = st.sidebar.selectbox("Select Numeric Column for Analysis (e.g., Height, Price):", 
                                         options=numeric_cols, 
                                         index=0 if numeric_cols else None)
    
    primary_category = st.sidebar.selectbox("Select Primary Category (e.g., Genre, Department):", 
                                           options=categorical_cols, 
                                           index=0 if categorical_cols else None)
    
    secondary_category = st.sidebar.selectbox("Select Secondary Category (e.g., Publisher, Region):", 
                                             options=categorical_cols, 
                                             index=min(1, len(categorical_cols)-1) if len(categorical_cols) > 1 else None)
    
    label_col = st.sidebar.selectbox("Select Text Label Column (e.g., Title, Item Name):", 
                                    options=all_cols, 
                                    index=0 if all_cols else None)

    # Global Data Filtering based on user selections
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Global Dashboard Filters")
    
    filtered_df = df.copy()
    
    if primary_category and filtered_df[primary_category].nunique() < 100:
        p_choices = st.sidebar.multiselect(f"Filter by {primary_category}:", 
                                           options=sorted(filtered_df[primary_category].dropna().unique().astype(str)),
                                           default=sorted(filtered_df[primary_category].dropna().unique().astype(str)))
        if p_choices:
            filtered_df = filtered_df[filtered_df[primary_category].astype(str).isin(p_choices)]

    # 5. Advanced Feature Engineering Engine
    # Dynamic Outlier Detection
    if target_numeric and primary_category:
        filtered_df['Is_Outlier'] = False
        for group in filtered_df[primary_category].dropna().unique():
            group_mask = filtered_df[primary_category] == group
            vals = filtered_df.loc[group_mask, target_numeric].dropna()
            if len(vals) > 3:
                q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
                iqr = q3 - q1
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                filtered_df.loc[group_mask, 'Is_Outlier'] = ~filtered_df.loc[group_mask, target_numeric].between(lower, upper)
    else:
        filtered_df['Is_Outlier'] = False

    # 6. Dashboard Tabs Interface
    tab1, tab2, tab3 = st.tabs(["📊 Executive Summary", "🔬 Advanced Cross-Correlations", "📋 Granular Data Explorer"])

    # ==========================================
    # TAB 1: EXECUTIVE SUMMARY
    # ==========================================
    with tab1:
        # Dynamic KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{filtered_df.shape[0]:,}")
        with col2:
            st.metric("Total Columns", f"{filtered_df.shape[1]:,}")
        with col3:
            if target_numeric:
                avg_val = filtered_df[target_numeric].mean()
                st.metric(f"Avg {target_numeric}", f"{avg_val:.2f}" if not pd.isna(avg_val) else "N/A")
            else:
                st.metric("Numeric Metric", "None Selected")
        with col4:
            outlier_count = filtered_df['Is_Outlier'].sum() if 'Is_Outlier' in filtered_df else 0
            st.metric("Statistical Outliers", outlier_count)
            
        st.markdown("---")
        
        # High Density Layout Row 1
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            st.markdown(f"### 🌲 Hierarchical Concentration Map")
            if primary_category and secondary_category:
                if primary_category != secondary_category:
                    fig_tree = px.treemap(filtered_df, path=[primary_category, secondary_category], 
                                          title=f"Dataset Volume Mix ({primary_category} ➔ {secondary_category})",
                                          template="plotly_white")
                    st.plotly_chart(fig_tree, use_container_width=True)
                else:
                    st.warning("Please select two different columns for Primary and Secondary categories to display the Treemap.")
            else:
                st.info("Select both a primary and secondary category in the sidebar to generate the Treemap.")
            
        with row1_col2:
            st.markdown(f"### 🍰 Share Distribution of {primary_category or 'Category'}")
            if primary_category:
                cat_counts = filtered_df[primary_category].value_counts().reset_index()
                cat_counts.columns = [primary_category, 'Count']
                fig_pie = px.pie(cat_counts, values='Count', names=primary_category, hole=0.4,
                                 color_discrete_sequence=px.colors.sequential.Blues,
                                 template="plotly_white")
                fig_pie.update_traces(textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("Awaiting Primary Category mapping setup.")

    # ==========================================
    # TAB 2: ADVANCED CROSS-CORRELATIONS
    # ==========================================
    with tab2:
        st.markdown("### 🔬 Strategic Visualizations & Statistical Breakdown")
        row2_col1, row2_col2 = st.columns(2)
        
        with row2_col1:
            st.markdown("### 📦 Structural Variances (Box Plot)")
            if primary_category and target_numeric:
                fig_box = px.box(filtered_df, x=primary_category, y=target_numeric, color=primary_category,
                                 points="outliers", title=f"Spread of {target_numeric} across {primary_category}",
                                 template="plotly_white")
                st.plotly_chart(fig_box, use_container_width=True)
            else:
                st.info("Requires a valid Numeric Metric and a Primary Category selection.")
            
        with row2_col2:
            st.markdown("### 🌡️ Cross-Tabulation Matrix (Heatmap)")
            if primary_category and secondary_category and label_col:
                # SAFE CHECK: Prevent the 1-dimensional grouper error if selections overlap
                if primary_category == secondary_category:
                    st.warning("⚠️ Heatmap generation skipped: 'Primary Category' and 'Secondary Category' dropdowns cannot point to the same column simultaneously.")
                elif not filtered_df.empty:
                    # Perform grouping securely using size() rather than pivot_table to avoid dimension index errors
                    pivot = filtered_df.groupby([primary_category, secondary_category]).size().unstack(fill_value=0)
                    if not pivot.empty:
                        fig_heat = px.imshow(pivot, text_auto=True, color_continuous_scale='Blues',
                                             labels=dict(x=secondary_category, y=primary_category, color="Count"),
                                             template="plotly_white")
                        st.plotly_chart(fig_heat, use_container_width=True)
                    else:
                        st.info("No combination matrix could be evaluated.")
                else:
                    st.info("Filtered dataset is empty.")
            else:
                st.info("Requires Primary Category, Secondary Category, and a Label column mapped.")

        st.markdown("---")
        
        # Dominance / Concentration Matrix Row
        row3_col1, row3_col2 = st.columns([1, 2])
        with row3_col1:
            st.markdown("### 👑 Volume Top Dominance")
            if primary_category:
                cr4 = filtered_df[primary_category].value_counts(normalize=True).head(4).sum() * 100
                st.progress(min(cr4 / 100, 1.0))
                st.caption(f"**Concentration Ratio (CR4):** The top 4 categories account for **{cr4:.1f}%** of all dataset entries.")
            else:
                st.caption("Awaiting categorization mapping.")
            
        with row3_col2:
            st.markdown("### 🔝 Leaderboard Matrix")
            if primary_category:
                top_items = filtered_df[primary_category].value_counts().head(10).reset_index()
                top_items.columns = [primary_category, 'Occurrences']
                st.dataframe(top_items, use_container_width=True, hide_index=True)

    # ==========================================
    # TAB 3: GRANULAR DATA EXPLORER
    # ==========================================
    with tab3:
        st.markdown("### 🔍 Live Text Search & Filtering")
        
        col_s1, col_s2, col_s3 = st.columns([2, 1, 1])
        with col_s1:
            search_term = st.text_input("Global string query filter:", placeholder="Type any search phrase...")
        with col_s2:
            only_show_outliers = st.checkbox("⚠️ Filter Down to Statistical Outliers")
        with col_s3:
            csv_buffer = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(label="📥 Export Current View to CSV", data=csv_buffer, file_name="filtered_export.csv", mime="text/csv")

        # Query filter process pipeline
        final_df = filtered_df.copy()
        
        if search_term and label_col:
            final_df = final_df[final_df[label_col].astype(str).str.contains(search_term, case=False, na=False)]
            
        if only_show_outliers and 'Is_Outlier' in final_df:
            final_df = final_df[final_df['Is_Outlier'] == True]

        # Final Render DataFrames
        st.dataframe(final_df, use_container_width=True)
        
        st.markdown("### 🧮 Numerical Structural Summary Table")
        st.write(final_df.describe().T)

else:
    # Landing page state when app is empty
    st.info("👈 Please upload a CSV file via the sidebar to initiate the dashboard application.")
    
    # Quick illustration panel
    st.markdown("""
    ### 🚀 Getting Started
    1. Drag and drop any `.csv` file into the left side pane.
    2. Map out the relevant **Numeric Metric**, **Primary Category**, and **Secondary Category** items in the dropdowns.
    3. The application will instantly adjust the text filters, calculate data concentration scales, and flag outliers matching your custom parameters.
    """)