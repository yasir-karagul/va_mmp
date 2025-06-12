import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_curve, roc_auc_score)
from sklearn.metrics import precision_score, recall_score, f1_score, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

try:
    train_data = pd.read_csv(r'sample_train.csv')
    print(f"Eğitim satir sayisi: {train_data.shape[0]} sutun sayisi {train_data.shape[1]}")
    print(train_data.columns)
except FileNotFoundError as e:
    print(f"Dosya bulunamadı: {e}")
    print("Örnek veri seti oluşturuluyor...")

# on islemleri
# reduce row size from 8.5m to 1k
sample_data = train_data.sample(n=1000, random_state=42)  # random_state for reproducibility

# null rows
print("\nMissing values in train:\n", sample_data.isnull().sum().sort_values(ascending=False).head(20))
cols_to_drop = ['DefaultBrowsersIdentifier', 'PuaMode', 'Census_ProcessorClass']
sample_data.drop(cols_to_drop, axis=1, inplace=True)  # update to original data set

print(f"silmekten sonra Eğitim satir sayisi: {sample_data.shape[0]} sutun sayisi {sample_data.shape[1]}")

X = sample_data.drop('HasDetections', axis=1)
y = sample_data['HasDetections']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle categorical features
categorical_cols = X_train.select_dtypes(include=['object']).columns
numerical_cols = X_train.select_dtypes(exclude=['object']).columns

print('0..........................................................................................0')
print(f' here : {categorical_cols}')
print(numerical_cols)

# filling missing data
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

X_train_num = pd.DataFrame(num_imputer.fit_transform(X_train[numerical_cols]),
                           columns=numerical_cols)  # fit_transform  fills in missing values only)
X_val_num = pd.DataFrame(num_imputer.transform(X_val[numerical_cols]), columns=numerical_cols)

X_train_cat = pd.DataFrame(cat_imputer.fit_transform(X_train[categorical_cols]), columns=categorical_cols)
X_val_cat = pd.DataFrame(cat_imputer.transform(X_val[categorical_cols]), columns=categorical_cols)

# Encode categorical variables - Handle unseen labels
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()

    # Fit on training data
    X_train_cat[col] = le.fit_transform(X_train_cat[col].astype(str))


    # Handle unseen labels in validation set
    def safe_transform(series, encoder):
        # Create a copy of the series
        series_copy = series.astype(str).copy()

        # Replace unseen labels with the most frequent class from training
        mask = ~series_copy.isin(encoder.classes_)
        if mask.any():
            most_frequent = encoder.classes_[0]  # Use first class as default
            series_copy[mask] = most_frequent
            print(f"Warning: Found {mask.sum()} unseen labels in column '{col}', replaced with '{most_frequent}'")

        return encoder.transform(series_copy)


    X_val_cat[col] = safe_transform(X_val_cat[col], le)
    label_encoders[col] = le

# Combine numerical and categorical features
X_train_processed = pd.concat([X_train_num, X_train_cat], axis=1)
X_val_processed = pd.concat([X_val_num, X_val_cat], axis=1)

# Scale numerical features
scaler = StandardScaler()
X_train_processed[numerical_cols] = scaler.fit_transform(X_train_processed[numerical_cols])
X_val_processed[numerical_cols] = scaler.transform(X_val_processed[numerical_cols])

# Check final shapes
print("Processed train shape:", X_train_processed.shape)
print("Processed validation shape:", X_val_processed.shape)

# =============================================================================
# K-MEANS CLUSTERING ANALYSIS
# =============================================================================

print("\n" + "=" * 80)
print("K-MEANS CLUSTERING ANALYSIS")
print("=" * 80)


# Find optimal number of clusters using elbow method
def find_optimal_clusters(X, max_k=10):
    """Find optimal number of clusters using elbow method and silhouette score"""
    inertias = []
    silhouette_scores = []
    K_range = range(2, max_k + 1)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, cluster_labels))

    return K_range, inertias, silhouette_scores


# Find optimal clusters
print("Finding optimal number of clusters...")
K_range, inertias, silhouette_scores = find_optimal_clusters(X_train_processed, max_k=8)

# Plot elbow curve and silhouette scores
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(K_range, silhouette_scores, 'ro-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs Number of Clusters')
plt.grid(True, alpha=0.3)

# Choose optimal k (highest silhouette score or elbow point)
optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"Optimal number of clusters based on silhouette score: {optimal_k}")

# Apply K-means with optimal k
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
train_clusters = kmeans.fit_predict(X_train_processed)
val_clusters = kmeans.predict(X_val_processed)

# Calculate silhouette score for optimal k
train_silhouette = silhouette_score(X_train_processed, train_clusters)
print(f"Training Silhouette Score with k={optimal_k}: {train_silhouette:.4f}")

# Analyze clusters
print(f"\nCluster Analysis:")
print(f"Cluster distribution in training set:")
unique, counts = np.unique(train_clusters, return_counts=True)
for cluster, count in zip(unique, counts):
    print(f"Cluster {cluster}: {count} samples ({count / len(train_clusters) * 100:.1f}%)")

# Analyze relationship between clusters and target variable
print(f"\nCluster vs Target Analysis:")
cluster_target_analysis = pd.crosstab(train_clusters, y_train, normalize='index')
print(cluster_target_analysis)

# Calculate malware detection rate per cluster
malware_rates = []
for cluster in range(optimal_k):
    cluster_mask = train_clusters == cluster
    malware_rate = y_train[cluster_mask].mean()
    malware_rates.append(malware_rate)
    print(f"Cluster {cluster}: {malware_rate:.3f} malware detection rate")

# Visualize cluster-target relationship
plt.subplot(1, 3, 3)
plt.bar(range(optimal_k), malware_rates)
plt.xlabel('Cluster')
plt.ylabel('Malware Detection Rate')
plt.title('Malware Detection Rate by Cluster')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Add cluster features for enhanced classification
print("\nAdding cluster features to dataset...")
X_train_with_clusters = X_train_processed.copy()
X_val_with_clusters = X_val_processed.copy()

X_train_with_clusters['cluster'] = train_clusters
X_val_with_clusters['cluster'] = val_clusters

# Also add distance to cluster centers as features
train_distances = kmeans.transform(X_train_processed)
val_distances = kmeans.transform(X_val_processed)

for i in range(optimal_k):
    X_train_with_clusters[f'dist_to_cluster_{i}'] = train_distances[:, i]
    X_val_with_clusters[f'dist_to_cluster_{i}'] = val_distances[:, i]

print(f"Enhanced dataset shape with clusters: {X_train_with_clusters.shape}")

# =============================================================================
# DECISION TREE CLASSIFICATION (Enhanced with Clusters)
# =============================================================================

print("\n" + "=" * 80)
print("DECISION TREE CLASSIFICATION (WITH CLUSTER FEATURES)")
print("=" * 80)

# Compare models with and without cluster features
models_comparison = {}

# Original model (without clusters)
print("Training Decision Tree without cluster features...")
dt_original = DecisionTreeClassifier(random_state=42)
dt_original.fit(X_train_processed, y_train)
pred_original = dt_original.predict(X_val_processed)
pred_proba_original = dt_original.predict_proba(X_val_processed)[:, 1]

models_comparison['Original'] = {
    'accuracy': accuracy_score(y_val, pred_original),
    'roc_auc': roc_auc_score(y_val, pred_proba_original)
}

# Enhanced model (with clusters)
print("Training Decision Tree with cluster features...")
dt_enhanced = DecisionTreeClassifier(random_state=42)
dt_enhanced.fit(X_train_with_clusters, y_train)
pred_enhanced = dt_enhanced.predict(X_val_with_clusters)
pred_proba_enhanced = dt_enhanced.predict_proba(X_val_with_clusters)[:, 1]

models_comparison['With Clusters'] = {
    'accuracy': accuracy_score(y_val, pred_enhanced),
    'roc_auc': roc_auc_score(y_val, pred_proba_enhanced)
}

# Calculate all metrics for both models
def calculate_comprehensive_metrics(y_true, y_pred, y_pred_proba, model_name):
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred),
        'ROC AUC': roc_auc_score(y_true, y_pred_proba),
        'Error Rate': 1 - accuracy_score(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred))
    }
    return metrics


# Calculate metrics for both models
original_metrics = calculate_comprehensive_metrics(y_val, pred_original, pred_proba_original,
                                                   'Decision Tree (Original)')
enhanced_metrics = calculate_comprehensive_metrics(y_val, pred_enhanced, pred_proba_enhanced, 'Decision Tree + K-Means')

# Create comparison DataFrame
comparison_df = pd.DataFrame([original_metrics, enhanced_metrics])

# Print comprehensive comparison table
print("\n" + "=" * 80)
print("COMPREHENSIVE NUMERICAL COMPARISON OF ALGORITHMS")
print("=" * 80)
print(comparison_df.round(4).to_string(index=False))

# Calculate improvements
print("\n" + "=" * 60)
print("PERFORMANCE IMPROVEMENTS WITH K-MEANS CLUSTERING")
print("=" * 60)

improvements = {
    'Accuracy': enhanced_metrics['Accuracy'] - original_metrics['Accuracy'],
    'Precision': enhanced_metrics['Precision'] - original_metrics['Precision'],
    'Recall': enhanced_metrics['Recall'] - original_metrics['Recall'],
    'F1-Score': enhanced_metrics['F1-Score'] - original_metrics['F1-Score'],
    'ROC AUC': enhanced_metrics['ROC AUC'] - original_metrics['ROC AUC'],
    'Error Rate': enhanced_metrics['Error Rate'] - original_metrics['Error Rate'],
    'MSE': enhanced_metrics['MSE'] - original_metrics['MSE'],
    'RMSE': enhanced_metrics['RMSE'] - original_metrics['RMSE']
}

for metric, improvement in improvements.items():
    direction = "↑" if metric not in ['Error Rate', 'MSE', 'RMSE'] else "↓"
    status = "BETTER" if (improvement > 0 and metric not in ['Error Rate', 'MSE', 'RMSE']) or (
                improvement < 0 and metric in ['Error Rate', 'MSE', 'RMSE']) else "WORSE"
    print(f"{metric:<12}: {improvement:+.4f} {direction} ({status})")

# Summary statistics
print(f"\n{'=' * 40}")
print("SUMMARY STATISTICS")
print(f"{'=' * 40}")
print(
    f"Best performing model: {'K-Means Enhanced' if enhanced_metrics['Accuracy'] > original_metrics['Accuracy'] else 'Original'}")
print(f"Accuracy improvement: {improvements['Accuracy']:+.4f}")
print(f"F1-Score improvement: {improvements['F1-Score']:+.4f}")
print(f"ROC AUC improvement: {improvements['ROC AUC']:+.4f}")
print(f"Error rate reduction: {-improvements['Error Rate']:+.4f}")

# Store for final comparison
models_comparison = {
    'Original': original_metrics,
    'With Clusters': enhanced_metrics,
    'Improvements': improvements
}

# Feature importance analysis for enhanced model
feature_importance_enhanced = pd.DataFrame({
    'feature': X_train_with_clusters.columns,
    'importance': dt_enhanced.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 15 Feature Importances (Enhanced Model):")
print(feature_importance_enhanced.head(15))

# Check importance of cluster-related features
cluster_features = [col for col in X_train_with_clusters.columns if 'cluster' in col or 'dist_to_cluster' in col]
cluster_importance = feature_importance_enhanced[feature_importance_enhanced['feature'].isin(cluster_features)]
print(f"\nCluster-related feature importances:")
print(cluster_importance)

# Visualization
plt.figure(figsize=(15, 10))

# Feature Importance comparison
plt.subplot(2, 3, 1)
top_features_enhanced = feature_importance_enhanced.head(12)
plt.barh(range(len(top_features_enhanced)), top_features_enhanced['importance'])
plt.yticks(range(len(top_features_enhanced)), top_features_enhanced['feature'])
plt.xlabel('Importance')
plt.title('Top 12 Feature Importances (Enhanced)')
plt.grid(True, alpha=0.3)

# ROC Curves comparison
plt.subplot(2, 3, 2)
fpr_orig, tpr_orig, _ = roc_curve(y_val, pred_proba_original)
fpr_enh, tpr_enh, _ = roc_curve(y_val, pred_proba_enhanced)

plt.plot(fpr_orig, tpr_orig, linewidth=2, label=f'Original (AUC = {original_metrics["ROC AUC"]:.3f})')
plt.plot(fpr_enh, tpr_enh, linewidth=2, label=f'With Clusters (AUC = {enhanced_metrics["ROC AUC"]:.3f})')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

# Confusion Matrix for enhanced model
plt.subplot(2, 3, 3)
cm_enhanced = confusion_matrix(y_val, pred_enhanced)
sns.heatmap(cm_enhanced, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Enhanced)')


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_processed)

plt.subplot(2, 3, 4)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=train_clusters, cmap='viridis', alpha=0.6)
plt.colorbar(scatter)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.title('Clusters in PCA Space')

# Cluster centers in PCA space
centers_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', marker='x', s=200, linewidths=3)

# Model accuracy comparison
plt.subplot(2, 3, 5)
models_names = ['Original', 'With Clusters']
accuracies = [original_metrics['Accuracy'], enhanced_metrics['Accuracy']]
plt.bar(models_names, accuracies)
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.ylim(0, 1)
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.01, f'{acc:.3f}', ha='center')

# Cluster feature importance
plt.subplot(2, 3, 6)
if len(cluster_importance) > 0:
    plt.barh(range(len(cluster_importance)), cluster_importance['importance'])
    plt.yticks(range(len(cluster_importance)), cluster_importance['feature'])
    plt.xlabel('Importance')
    plt.title('Cluster Feature Importances')
    plt.grid(True, alpha=0.3)
else:
    plt.text(0.5, 0.5, 'No significant cluster features', ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Cluster Feature Importances')

plt.tight_layout()
plt.show()

# =============================================================================
# ORIGINAL DECISION TREE CLASSIFICATION (for comparison)
# ============================================================================

print("\n" + "=" * 80)
print("ORIGINAL DECISION TREE CLASSIFICATION RESULTS")
print("=" * 80)

print(f"Decision Tree Results (Original):")
print(f"Accuracy: {original_metrics['Accuracy']:.4f}")
print(f"ROC AUC: {original_metrics['ROC AUC']:.4f}")
print(f"Tree Depth: {dt_original.tree_.max_depth}")
print(f"Number of Leaves: {dt_original.tree_.n_leaves}")

print(f"Decision Tree Results (Enhanced with Clusters):")
print(f"Accuracy: {enhanced_metrics['Accuracy']:.4f}")
print(f"ROC AUC: {enhanced_metrics['ROC AUC']:.4f}")
print(f"Tree Depth: {dt_enhanced.tree_.max_depth}")
print(f"Number of Leaves: {dt_enhanced.tree_.n_leaves}")

# Detailed evaluation for enhanced model
print(f"\nClassification Report (Enhanced Model):")
print(classification_report(y_val, pred_enhanced))

# Additional algorithm comparison with K-Means clustering
print("\n" + "=" * 80)
print("COMPARISON WITH OTHER ALGORITHMS (WITH K-MEANS FEATURES)")
print("=" * 80)

# Test multiple algorithms with cluster-enhanced features
algorithms = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'SVM (RBF)': SVC(probability=True, random_state=42)
}

# Store results for all algorithms
all_results = []

print("Training and evaluating multiple algorithms...")
for name, algorithm in algorithms.items():
    print(f"Training {name}...")

    # Train algorithm
    algorithm.fit(X_train_with_clusters, y_train)

    # Predictions
    pred = algorithm.predict(X_val_with_clusters)
    pred_proba = algorithm.predict_proba(X_val_with_clusters)[:, 1]

    # Calculate metrics
    metrics = calculate_comprehensive_metrics(y_val, pred, pred_proba, name)
    all_results.append(metrics)

# Create comprehensive results DataFrame
results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('Accuracy', ascending=False)

print("\n" + "=" * 100)
print("FINAL NUMERICAL COMPARISON OF ALL ALGORITHMS (WITH K-MEANS CLUSTERING)")
print("=" * 100)
print(results_df.round(4).to_string(index=False))

# Find best performing algorithm
best_algorithm = results_df.iloc[0]['Model']
best_accuracy = results_df.iloc[0]['Accuracy']
best_f1 = results_df.iloc[0]['F1-Score']
best_auc = results_df.iloc[0]['ROC AUC']

print(f"\n{'=' * 60}")
print("BEST PERFORMING ALGORITHM")
print(f"{'=' * 60}")
print(f"Algorithm: {best_algorithm}")
print(f"Accuracy: {best_accuracy:.4f}")
print(f"F1-Score: {best_f1:.4f}")
print(f"ROC AUC: {best_auc:.4f}")

# Performance ranking
print(f"\n{'=' * 40}")
print("ALGORITHM RANKING BY ACCURACY")
print(f"{'=' * 40}")
for i, row in results_df.iterrows():
    print(f"{row.name + 1:2d}. {row['Model']:<20}: {row['Accuracy']:.4f}")

print("\nK-Means Enhanced Multi-Algorithm Analysis Completed!")
print(f"Best Algorithm: {best_algorithm} with {best_accuracy:.4f} accuracy")

# K-Means clustering insights
print(f"\n{'=' * 60}")
print("K-MEANS CLUSTERING INSIGHTS")
print(f"{'=' * 60}")
print(f"Optimal number of clusters: {optimal_k}")
print(f"Silhouette score: {train_silhouette:.4f}")
print(f"Cluster features added: {len(cluster_features)}")
print("Malware distribution by cluster:")
for i, rate in enumerate(malware_rates):
    print(f"  Cluster {i}: {rate:.3f} malware rate ({rate * 100:.1f}%)")

if any(rate > 0.8 for rate in malware_rates):
    print("High-risk clusters detected (>80% malware rate)")
elif any(rate > 0.6 for rate in malware_rates):
    print("Medium-risk clusters detected (>60% malware rate)")
else:
    print("No high-risk clusters detected")
