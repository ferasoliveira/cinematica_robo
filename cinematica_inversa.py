import numpy as np
import matplotlib.pyplot as plt

# === Valores fixos dos elos ===
d1, a2, a3, a4, a5 = 1, 1, 1, 1, 1

# === Limites das juntas (rad) ===
limites = {
    0: (0, np.pi),             # theta1
    1: (np.deg2rad(15), np.deg2rad(75)),  # theta2
    2: (np.deg2rad(15), np.deg2rad(75)),  # theta3
    3: (np.deg2rad(15), np.deg2rad(75)),  # theta4
    4: (0, 2*np.pi)            # theta5
}

# === Função para matriz DH em NumPy ===
def dh_matrix_np(theta, d, a, alpha):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca, st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,    sa,    ca,    d],
        [0,    0,     0,     1]
    ])

# === Calcula posição do end-effector ===
def calcular_end_effector(thetas):
    t1, t2, t3, t4, t5 = thetas
    A1 = dh_matrix_np(t1, d1, 0, np.pi/2)
    A2 = dh_matrix_np(t2, 0, a2, 0)
    A3 = dh_matrix_np(t3, 0, a3, 0)
    A4 = dh_matrix_np(t4, 0, a4, 0)
    A5 = dh_matrix_np(t5, 0, a5, -np.pi/2)
    T = A1 @ A2 @ A3 @ A4 @ A5
    return T[:3,3]

# === Calcula posições de todos os elos para plot ===
def calcular_pontos(thetas):
    t1, t2, t3, t4, t5 = thetas
    A1 = dh_matrix_np(t1, d1, 0, np.pi/2)
    A2 = dh_matrix_np(t2, 0, a2, 0)
    A3 = dh_matrix_np(t3, 0, a3, 0)
    A4 = dh_matrix_np(t4, 0, a4, 0)
    A5 = dh_matrix_np(t5, 0, a5, -np.pi/2)
    T1 = A1
    T2 = A1 @ A2
    T3 = T2 @ A3
    T4 = T3 @ A4
    T5 = T4 @ A5
    pontos = np.array([
        [0,0,0],
        T1[:3,3],
        T2[:3,3],
        T3[:3,3],
        T4[:3,3],
        T5[:3,3]
    ])
    return pontos

# === Jacobiano numérico 3x5 ===
def jacobiano_numerico(thetas, delta=1e-5):
    J = np.zeros((3,5))
    f0 = calcular_end_effector(thetas)
    for i in range(5):
        thetas_d = thetas.copy()
        thetas_d[i] += delta
        f1 = calcular_end_effector(thetas_d)
        J[:,i] = (f1 - f0)/delta
    return J

# === IK iterativa com limites ===
def cinematica_inversa_iterativa(p_des, max_iter=300, tol=1e-4, alpha=0.5):
    thetas = np.zeros(5)  # chute inicial
    for _ in range(max_iter):
        p_curr = calcular_end_effector(thetas)
        e = p_des - p_curr
        if np.linalg.norm(e) < tol:
            break
        J = jacobiano_numerico(thetas)
        dtheta = alpha * np.linalg.pinv(J) @ e
        thetas += dtheta
        # aplica limites
        for i in range(5):
            thetas[i] = np.clip(thetas[i], limites[i][0], limites[i][1])
    return thetas

# === Gera coordenada aleatória dentro do alcance do robô ===
def gerar_ponto_aleatorio():
    l_total = a2 + a3 + a4 + a5
    r = np.random.uniform(0.1, l_total)
    theta = np.random.uniform(0, 2*np.pi)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.random.uniform(d1, d1 + l_total)
    return np.array([x, y, z])

# === Executa exemplo ===
p_d = gerar_ponto_aleatorio()
thetas_sol = cinematica_inversa_iterativa(p_d)
p_final = calcular_end_effector(thetas_sol)

print("Ponto aleatório desejado:", p_d)
print("Posição alcançada:", p_final)
print("Juntas calculadas (rad):", thetas_sol)

# === Plot 3D ===
pontos = calcular_pontos(thetas_sol)
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
cores = ['r','g','b','m','c']
for i in range(5):
    ax.plot(pontos[i:i+2,0], pontos[i:i+2,1], pontos[i:i+2,2], c=cores[i], lw=3)
ax.scatter(p_d[0], p_d[1], p_d[2], c='k', marker='o', s=50, label='End-effector desejado')
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
ax.set_title("Braço Robótico 5R - Posição Aleatória com Limites")
ax.set_box_aspect([1,1,1])
ax.set_xlim(-5,5); ax.set_ylim(-5,5); ax.set_zlim(0,5)
ax.legend()
plt.show()
