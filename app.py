import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import umap
import plotly.express as px

def load_data(file):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.name.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file)
    elif file.name.endswith('.tsv'):
        df = pd.read_csv(file, sep='\t')
    else:
        st.error("Unsupported file format. Please upload a CSV, Excel, or TSV file.")
        return None
    return df

def perform_pca(df, n_components):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df.select_dtypes(include=[np.number]))
    try:
        pca = PCA(n_components=n_components)
        return pca.fit_transform(scaled_data)
    except ValueError as e:
        if "must be between 0 and min(n_samples, n_features)" in str(e):
            return None
        raise e

def perform_umap(df, n_components):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df.select_dtypes(include=[np.number]))
    reducer = umap.UMAP(n_components=n_components)
    return reducer.fit_transform(scaled_data)

def perform_feature_selection(X, y, k):
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    return X_new, selected_features

def perform_classification(X, y, algorithm, param):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if algorithm == "KNN":
        clf = KNeighborsClassifier(n_neighbors=param)
    else:
        clf = RandomForestClassifier(n_estimators=param, random_state=42)
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    if len(np.unique(y)) == 2:
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    else:
        y_pred_proba = clf.predict_proba(X_test)
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
    
    return accuracy, f1, roc_auc

def main():
    st.title("Data Mining and Analysis Application")

    with st.sidebar:
        selected = option_menu(
            menu_title="Main Menu",
            options=["Dataset Preview", "Visualization", "Machine Learning", "Info"],
            icons=["table", "graph-up", "cpu", "info-circle-fill"],
            menu_icon="cast",
            default_index=0
        )
        
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls", "tsv"])

    if selected == "Info":
        with st.container(border=True):
            st.header("Σχετικά με την εφαρμογή")
            st.write("""
            Η web εφαρμογή που έχουμε σχεδιάσει παρέχει στους χρήστες εργαλεία για την εκτέλεση εργασιών μηχανικής μάθησης. Οι χρήστες μπορούν να αναρτήσουν τα δεδομένα, να τα oπτικοποιήσουν και να εφαρμόσουν αλγορίθμους κατηγοριοποίησης.
            """)
            st.header("Πως δουλεύει")
            st.write("""
            1. **Data Upload**: Ξεκινήστε με τη μεταφόρτωση του συνόλου δεδομένων σας (μορφή CSV, Excel ή TSV).

            2. **Dataset Preview**: Προβάλετε τα δεδομένα σας σε μορφή πίνακα για να αποκτήσετε μια αρχική εικόνα της δομής τους.

            3. **Visualization**:
                - **Dimensionality Reduction**: Χρησιμοποιήστε PCA ή UMAP για να απεικονίσετε τα δεδομένα σας σε 2D και 3D.
                - **Exploratory Data Analysis**: Δημιουργήστε histograms και scatter plots για να διερευνήσετε τις σχέσεις στα δεδομένα σας.

            4. **Machine Learning**:
                - **Feature Selection**: Μειώστε τον αριθμό των χαρακτηριστικών στο σύνολο δεδομένων σας για να επικεντρωθείτε στα πιο σημαντικά.
                - **Classification**: Εφαρμόστε αλγορίθμους KNN ή Random Forest στα δεδομένα σας, συγκρίνοντας τις επιδόσεις πριν και μετά την επιλογή χαρακτηριστικών.
            """)
            st.header("Ανάπτυξη")
            st.write("""
            Η ανάπτυξη της web εφαρμογής έγινε ολοκληρωτικά από τον Ιωάννη Μπαμπλέκη (inf2021153)
            """)

    elif uploaded_file is None:
        st.info("Please upload a file to continue.")

    else:
        df = load_data(uploaded_file)
        
        if df is not None:
            if selected == "Dataset Preview":
                st.subheader("Dataset Preview")
                st.dataframe(df, height=400)

            elif selected == "Visualization":
                st.subheader("Visualization")
                
                tab1, tab2 = st.tabs(["Dimensionality Reduction", "Exploratory Data Analysis"])
                
                with tab1:
                    st.subheader("Dimensionality Reduction Visualization")
                    dim_red_method = st.selectbox("Select dimensionality reduction method", ["PCA", "UMAP"])
                    dim = st.selectbox("Select dimension", ["2D", "3D"])
                    
                    dim = 2 if dim == "2D" else 3
                    if dim_red_method == "PCA":
                        reduced_data = perform_pca(df, dim)
                        if reduced_data is None:
                            st.error("Unable to perform PCA. The number of components must be less than or equal to the number of features in your dataset.")
                            st.stop()
                        color_scale = "viridis"
                    else:
                        reduced_data = perform_umap(df, dim)
                        color_scale = "plasma"
                    
                    if dim == 2:
                        fig = px.scatter(x=reduced_data[:, 0], y=reduced_data[:, 1],
                                         labels={'x': f'{dim_red_method} 1', 'y': f'{dim_red_method} 2'},
                                         color=reduced_data[:, 0], color_continuous_scale=color_scale)
                    else:
                        fig = px.scatter_3d(x=reduced_data[:, 0], y=reduced_data[:, 1], z=reduced_data[:, 2],
                                            labels={'x': f'{dim_red_method} 1', 'y': f'{dim_red_method} 2', 'z': f'{dim_red_method} 3'},
                                            color=reduced_data[:, 0], color_continuous_scale=color_scale)
                    
                    fig.update_layout(coloraxis_colorbar=dict(title=f'{dim_red_method} 1'))
                    st.plotly_chart(fig)
                
                with tab2:
                    st.subheader("Exploratory Data Analysis")
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    
                    plot_type = st.selectbox("Select plot type", ["Histogram", "Scatter Plot"])
                    
                    if plot_type == "Histogram":
                        hist_col = st.selectbox("Select column for histogram", numeric_cols)
                        fig_hist = px.histogram(df, x=hist_col, color_discrete_sequence=['#FFA07A'])
                        fig_hist.update_layout(title=f"Histogram of {hist_col}")
                        st.plotly_chart(fig_hist)
                    
                    else:
                        st.subheader("Scatter Plot")
                        x_col = st.selectbox("Select X-axis", numeric_cols)
                        y_col = st.selectbox("Select Y-axis", numeric_cols)
                        fig_scatter = px.scatter(df, x=x_col, y=y_col, color_discrete_sequence=['#20B2AA'])
                        fig_scatter.update_layout(title=f"Scatter Plot: {x_col} vs {y_col}")
                        st.plotly_chart(fig_scatter)

            elif selected == "Machine Learning":
                st.subheader("Machine Learning")
                
                tab1, tab2 = st.tabs(["Feature Selection", "Classification"])
                
                with tab1:
                    X = df.iloc[:, :-1]
                    y = df.iloc[:, -1]
                    
                    num_features = X.shape[1]
                    
                    if num_features > 1:
                        n_features = st.slider("Select number of features to keep", 
                                               min_value=1, max_value=num_features, value=min(5, num_features))
                        
                        X_new, selected_features = perform_feature_selection(X, y, n_features)
                        
                        st.write(f"Selected {n_features} features:")
                        st.write(selected_features)
                        
                        st.session_state.selected_features = selected_features
                    else:
                        st.warning("Your dataset has only one feature. Feature selection is not applicable.")
                        st.session_state.selected_features = X.columns.tolist()
                
                with tab2:
                    st.subheader("Classification")
                    
                    X = df.iloc[:, :-1]
                    y = df.iloc[:, -1]
                    
                    selected_features = st.session_state.get('selected_features', X.columns.tolist())
                    
                    X_original = X
                    X_reduced = X[selected_features]
                    
                    if len(selected_features) > 1:
                        algorithm = st.selectbox("Select classification algorithm", ["KNN", "Random Forest"])
                        
                        if algorithm == "KNN":
                            param = st.slider("Select number of neighbors (k)", min_value=1, max_value=20, value=5)
                        else:
                            param = st.slider("Select number of trees", min_value=10, max_value=200, value=100)
                        
                        if st.button("Run Classification"):
                            acc_orig, f1_orig, roc_auc_orig = perform_classification(X_original, y, algorithm, param)
                            
                            acc_red, f1_red, roc_auc_red = perform_classification(X_reduced, y, algorithm, param)
                            
                            st.subheader("Classification Results")
                            results = pd.DataFrame({
                                "Metric": ["Accuracy", "F1-Score", "ROC-AUC"],
                                "Original Dataset": [acc_orig, f1_orig, roc_auc_orig],
                                "Reduced Dataset": [acc_red, f1_red, roc_auc_red]
                            })
                            st.table(results.set_index("Metric"))
                            
                            st.subheader("Performance Comparison")
                            fig = px.bar(results, x="Metric", y=["Original Dataset", "Reduced Dataset"], 
                                         barmode="group", height=400)
                            st.plotly_chart(fig)
                    else:
                        st.warning("Classification requires at least two features. Please upload a dataset with more features or adjust your feature selection.")

if __name__ == "__main__":
    main()
