import numpy as np
import trimesh
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# =========================
# 1. Cargar modelo
# =========================
mesh = trimesh.load("modelo.ply")

# Guardar original directamente
mesh.export("original.obj")

# =========================
# 2. Voxelización
# =========================
voxel_size = mesh.scale / 50  # resolución (ajustable)
vox = mesh.voxelized(voxel_size)

# Obtener matriz binaria y coordenadas
matrix = vox.matrix
points = np.argwhere(matrix == True)

# Convertir a coordenadas reales
points = points * voxel_size

# =========================
# 3. Centrar objeto
# =========================
centroid = np.mean(points, axis=0)
points_centered = points - centroid

# =========================
# 4. Tensor de inercia
# =========================
I = np.zeros((3, 3))

for x, y, z in points_centered:
    I[0, 0] += y**2 + z**2
    I[1, 1] += x**2 + z**2
    I[2, 2] += x**2 + y**2
    I[0, 1] -= x * y
    I[0, 2] -= x * z
    I[1, 2] -= y * z

# Simetría
I[1, 0] = I[0, 1]
I[2, 0] = I[0, 2]
I[2, 1] = I[1, 2]

print("Tensor de inercia:\n", I)

# =========================
# 5. Eigenvalores y eigenvectores
# =========================
eigvals, eigvecs = eigh(I)

print("Eigenvalores:\n", eigvals)
print("Eigenvectores:\n", eigvecs)

# =========================
# 6. Matriz de rotación
# =========================
R = eigvecs  # columnas son los ejes principales

# =========================
# 7. Alinear puntos
# =========================
points_aligned = points_centered @ R

# =========================
# 7A. Visualización
# =========================
fig = plt.figure(figsize=(12, 6))

# Original
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(points_centered[:, 0],
            points_centered[:, 1],
            points_centered[:, 2],
            s=1)
ax1.set_title("Original")

# Alineado
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(points_aligned[:, 0],
            points_aligned[:, 1],
            points_aligned[:, 2],
            s=1)
ax2.set_title("Alineado")

plt.show()

# =========================
# 7B. Exportar mesh alineado
# =========================
# Centrar mesh original
mesh_centered = mesh.copy()
mesh_centered.vertices -= mesh_centered.vertices.mean(axis=0)

# Aplicar rotación
mesh_centered.vertices = mesh_centered.vertices @ R

# Guardar resultado
mesh_centered.export("aligned.obj")

print("Exportación completa: original.obj y aligned.obj")