import numpy as np, random, time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.primitives import Sampler
from azure.quantum.qiskit import AzureQuantumProvider

# === Set seeds for reproducibility ===
np.random.seed(42)
random.seed(42)

# === Load angle-encoded dataset ===
X = np.loadtxt("quantum_encoded_features.csv", delimiter=",")
y = pd.read_csv("quantum_labels.csv")["label"].map({"benign": 0, "attack": 1}).values

# Downsample for budget control (10% of dataset)
X, _, y, _ = train_test_split(X, y, train_size=0.1, stratify=y, random_state=42)

# === Classical PCA for Comparison ===
pca = PCA(n_components=10)
pca.fit(X)
explained_variance = pca.explained_variance_ratio_

# Save elbow curve
plt.figure()
plt.plot(range(1, 11), explained_variance[:10], marker='o')
plt.title("Classical PCA - Elbow Curve")
plt.xlabel("Components")
plt.ylabel("Explained Variance")
plt.grid(True)
plt.savefig("classical_pca_elbow.png")
print("📉 Saved classical_pca_elbow.png")

# === Azure Quantum Workspace (Real QPU) ===
provider = AzureQuantumProvider(
    resource_id="/subscriptions/9618f3b2-2d8b-45f9-ad9a-8265215539e3/resourceGroups/AzureQuantum/providers/Microsoft.Quantum/workspaces/qsoc-cli",
    location="westeurope"
)

backend = provider.get_backend("quantinuum.qpu.h1-1e")
sampler = Sampler()
sampler.set_options(default_backend=backend)

# === Use 3 qubits for cost efficiency ===
X_subset = X[:, :3]
feature_map = ZZFeatureMap(feature_dimension=3, reps=1)

qkernel = FidelityQuantumKernel(
    feature_map=feature_map,
    sampler=sampler
)

# === Run quantum kernel matrix evaluation ===
start = time.time()
kernel_matrix = qkernel.evaluate(x_vec=X_subset)
end = time.time()

# Save kernel matrix
np.savetxt("quantum_kernel_matrix.csv", kernel_matrix)

# === Quantum PCA on Kernel ===
kpca = KernelPCA(kernel="precomputed", n_components=3)
X_pca = kpca.fit_transform(kernel_matrix)

# Save 3D projection
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap="coolwarm", alpha=0.8)
plt.title("Quantum Kernel PCA - Real QPU")
plt.savefig("quantum_pca_projection_real.png")

# === Summary Output ===
print("✅ Real QPU Kernel matrix computed.")
print("📊 Classical PCA variance (Top 3):", explained_variance[:3])
print("🧪 Quantum PCA shape:", X_pca.shape)
print(f"⏱️ Time taken: {round(end - start, 2)} seconds")
