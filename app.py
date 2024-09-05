import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

st.title("Data Mining and Analysis")

with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Dataset Preview", "2D Visualization", "Machine Learning", "Info"],
        icons=["table", "graph-up", "cpu", "info-circle-fill"],
        menu_icon="cast",
        default_index=0
    )

uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "tsv"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file, encoding='latin1')
        elif uploaded_file.name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_file, encoding='latin1')
        elif uploaded_file.name.endswith('.tsv'):
            data = pd.read_csv(uploaded_file, delimiter='\t', encoding='latin1')

        # Check if the dataset contains any numeric data
        numeric_data = data.select_dtypes(include=['int64', 'float64'])

        if numeric_data.empty:
            st.error("The uploaded dataset does not contain any numeric data. Please upload a dataset with numeric values for analysis.")
        else:
            st.session_state.data = data

    except Exception as e:
        st.error(f"Error: {e}")

if selected == "Dataset Preview":
    if uploaded_file is not None:
        if 'data' in st.session_state:
            data = st.session_state.data
            st.success("Data Loaded Successfully.")
            st.dataframe(data)
        else:
            st.info("No data found in session state.")
    else:
        st.info("Please upload a CSV, Excel or TSV file to get started.")

elif selected == "2D Visualization":
    if uploaded_file is not None:
        if 'data' in st.session_state:
            data = st.session_state.data
            
            # Dropping non-numeric columns for PCA and UMAP
            numeric_data = data.select_dtypes(include=['int64', 'float64'])

            if numeric_data.shape[0] < 2 or numeric_data.shape[1] < 2:
                st.error("Not enough data or features to perform PCA or UMAP. Please upload a dataset with at least 2 samples and 2 numeric features.")
            else:
                # Standardizing the data
                scaler = StandardScaler()
                standardized_data = scaler.fit_transform(numeric_data)

                tab1, tab2 = st.tabs(["PCA & UMAP", "EDA"])

                with tab1:
                    method = st.selectbox("Select dimensionality reduction method:", ["PCA", "UMAP"])
                    if method == "PCA":
                        # Perform PCA
                        pca = PCA(n_components=2)
                        pca_result = pca.fit_transform(standardized_data)

                        plt.figure(figsize=(10, 7))
                        plt.scatter(pca_result[:, 0], pca_result[:, 1], c='blue', alpha=0.7)
                        plt.title('PCA - 2D Projection')
                        plt.xlabel('Principal Component 1')
                        plt.ylabel('Principal Component 2')
                        plt.grid(True)
                        st.pyplot(plt)

                    elif method == "UMAP":
                        # Perform UMAP
                        umap_model = umap.UMAP(n_components=2)
                        umap_result = umap_model.fit_transform(standardized_data)

                        plt.figure(figsize=(10, 7))
                        plt.scatter(umap_result[:, 0], umap_result[:, 1], c='red', alpha=0.7)
                        plt.title('UMAP - 2D Projection')
                        plt.xlabel('UMAP Component 1')
                        plt.ylabel('UMAP Component 2')
                        plt.grid(True)
                        st.pyplot(plt)

                with tab2:
                    eda_method = st.selectbox("Select EDA method:", ["Histogram", "Scatter Plot"])
                    
                    if eda_method == "Histogram":
                        column = st.selectbox("Select column for Histogram:", numeric_data.columns)
                        plt.figure(figsize=(10, 7))
                        plt.hist(data[column], bins=20, color='purple', alpha=0.7)
                        plt.title(f'Histogram of {column}')
                        plt.xlabel(column)
                        plt.ylabel('Frequency')
                        plt.grid(True)
                        st.pyplot(plt)

                    elif eda_method == "Scatter Plot":
                        columns = st.multiselect("Select columns for Scatter Plot (x, y):", numeric_data.columns, default=numeric_data.columns[:2])
                        if len(columns) == 2:
                            plt.figure(figsize=(10, 7))
                            plt.scatter(data[columns[0]], data[columns[1]], color='orange', alpha=0.7)
                            plt.title(f'Scatter Plot of {columns[0]} vs {columns[1]}')
                            plt.xlabel(columns[0])
                            plt.ylabel(columns[1])
                            plt.grid(True)
                            st.pyplot(plt)
                        else:
                            st.warning("Please select exactly two columns for the scatter plot.")
        
        else:
            st.info("No data found in session state. Please upload a file first.")
    else:
        st.info("Please upload a CSV, Excel or TSV file to get started.")
