import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# === Definição das variáveis simbólicas ===
theta1, theta2, theta3, theta4, theta5 = sp.symbols('theta1 theta2 theta3 theta4 theta5')
d1, a2, a3, a4, a5 = sp.symbols('d1 a2 a3 a4 a5')

# === Função para matriz DH ===
def dh_matrix(theta, d, a, alpha):
    return sp.Matrix([
        [sp.cos(theta), -sp.sin(theta)*sp.cos(alpha),  sp.sin(theta)*sp.sin(alpha), a*sp.cos(theta)],
        [sp.sin(theta),  sp.cos(theta)*sp.cos(alpha), -sp.cos(theta)*sp.sin(alpha), a*sp.sin(theta)],
        [0,              sp.sin(alpha),               sp.cos(alpha),               d],
        [0, 0, 0, 1]
    ])

# === Matrizes DH para cada junta ===
A1 = dh_matrix(theta1, d1, 0, sp.pi/2)
A2 = dh_matrix(theta2, 0, a2, 0)
A3 = dh_matrix(theta3, 0, a3, 0)
A4 = dh_matrix(theta4, 0, a4, 0)
A5 = dh_matrix(theta5, 0, a5, -sp.pi/2)

# === Valores fixos dos comprimentos dos elos ===
valores_fixos = {d1:1, a2:1, a3:1, a4:1, a5:1}

# === Função para calcular posições dos elos ===
def calcular_pontos(t1, t2, t3, t4, t5):
    subs = {**valores_fixos, theta1:t1, theta2:t2, theta3:t3, theta4:t4, theta5:t5}
    T1 = A1.evalf(subs=subs)
    T2 = (A1*A2).evalf(subs=subs)
    T3 = (A1*A2*A3).evalf(subs=subs)
    T4 = (A1*A2*A3*A4).evalf(subs=subs)
    T5 = (A1*A2*A3*A4*A5).evalf(subs=subs)
    
    pontos = np.array([
        [0,0,0],
        [float(T1[0,3]), float(T1[1,3]), float(T1[2,3])],
        [float(T2[0,3]), float(T2[1,3]), float(T2[2,3])],
        [float(T3[0,3]), float(T3[1,3]), float(T3[2,3])],
        [float(T4[0,3]), float(T4[1,3]), float(T4[2,3])],
        [float(T5[0,3]), float(T5[1,3]), float(T5[2,3])]
    ])
    return pontos

# === Declaração explícita dos ângulos das juntas ===
t1 = np.pi/4
t2 = np.pi/6
t3 = -np.pi/6
t4 = np.pi/8
t5 = -np.pi/2

# === Calcula posições dos elos ===
pontos = calcular_pontos(t1, t2, t3, t4, t5)

# === Print das coordenadas finais ===
print("Coordenadas finais do end-effector:")
print(f"X = {pontos[-1,0]:.3f}, Y = {pontos[-1,1]:.3f}, Z = {pontos[-1,2]:.3f}")

# === Plot 3D do robô ===
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
cores = ['r','g','b','m','c']

for i in range(5):
    ax.plot(pontos[i:i+2,0], pontos[i:i+2,1], pontos[i:i+2,2], c=cores[i], lw=3)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Braço Robótico 5R - Posição Específica")
ax.set_box_aspect([1,1,1])
ax.set_xlim(-4,4)
ax.set_ylim(-4,4)
ax.set_zlim(0,4)
plt.show()
