import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# === Definição das variáveis ===
theta1, theta2, theta3, theta4, theta5 = sp.symbols('theta1 theta2 theta3 theta4 theta5')
d1, a2, a3, a4, a5 = sp.symbols('d1 a2 a3 a4 a5')

# === Função para matriz DH padrão ===
def dh_matrix(theta, d, a, alpha):
    return sp.Matrix([
        [sp.cos(theta), -sp.sin(theta)*sp.cos(alpha),  sp.sin(theta)*sp.sin(alpha), a*sp.cos(theta)],
        [sp.sin(theta),  sp.cos(theta)*sp.cos(alpha), -sp.cos(theta)*sp.sin(alpha), a*sp.sin(theta)],
        [0,              sp.sin(alpha),               sp.cos(alpha),               d],
        [0, 0, 0, 1]
    ])

# === DH ajustada para 5 juntas ===
A1 = dh_matrix(theta1, d1, 0, sp.pi/2)  # Base: rotação Z
A2 = dh_matrix(theta2, 0, a2, 0)        # Junta 2: rotação X
A3 = dh_matrix(theta3, 0, a3, 0)        # Junta 3: rotação X
A4 = dh_matrix(theta4, 0, a4, 0)        # Junta 4: rotação X
A5 = dh_matrix(theta5, 0, a5, -sp.pi/2) # Junta 5: rotação Y (DH equivalente)

# === Valores fixos ===
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
        [0, 0, 0],
        [float(T1[0,3]), float(T1[1,3]), float(T1[2,3])],
        [float(T2[0,3]), float(T2[1,3]), float(T2[2,3])],
        [float(T3[0,3]), float(T3[1,3]), float(T3[2,3])],
        [float(T4[0,3]), float(T4[1,3]), float(T4[2,3])],
        [float(T5[0,3]), float(T5[1,3]), float(T5[2,3])]
    ])
    return pontos

# === Configuração inicial do gráfico ===
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.35)

cores = ['r','g','b','m','c']

# Posição inicial
pontos = calcular_pontos(0,0,0,0,0)
linhas = [ax.plot(pontos[i:i+2,0], pontos[i:i+2,1], pontos[i:i+2,2], c=cores[i], lw=3)[0] for i in range(5)]

# Configurações do gráfico
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Braço robótico 5R interativo")
ax.set_box_aspect([1,1,1])
ax.set_xlim(-4,4)
ax.set_ylim(-4,4)
ax.set_zlim(0,4)

# === Sliders ===
axcolor = 'lightgoldenrodyellow'
ax_theta1 = plt.axes([0.15, 0.25, 0.65, 0.03], facecolor=axcolor)
ax_theta2 = plt.axes([0.15, 0.20, 0.65, 0.03], facecolor=axcolor)
ax_theta3 = plt.axes([0.15, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_theta4 = plt.axes([0.15, 0.10, 0.65, 0.03], facecolor=axcolor)
ax_theta5 = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor=axcolor)

slider1 = Slider(ax_theta1, 'θ1', -np.pi, np.pi, valinit=0)
slider2 = Slider(ax_theta2, 'θ2', 0, np.pi/2, valinit=0)
slider3 = Slider(ax_theta3, 'θ3', -np.pi/2, 0, valinit=0)
slider4 = Slider(ax_theta4, 'θ4', -np.pi/2, 0, valinit=0)
slider5 = Slider(ax_theta5, 'θ5', -np.pi, np.pi, valinit=0)

# === Gráfico de X e Z da ponta ===
fig2, ax2 = plt.subplots()
ax2.set_xlabel("θ5 [rad]")
ax2.set_ylabel("Coordenadas")
linhaX, = ax2.plot([], [], 'r', label='X ponta')
linhaZ, = ax2.plot([], [], 'b', label='Z ponta')
ax2.legend()
ax2.set_xlim(-np.pi, np.pi)
ax2.set_ylim(-3,3)

theta5_vals = np.linspace(-np.pi, np.pi, 200)
x_vals = []
z_vals = []

# === Função de atualização ===
def update(val):
    t1 = slider1.val
    t2 = slider2.val
    t3 = slider3.val
    t4 = slider4.val
    t5 = slider5.val
    pontos = calcular_pontos(t1,t2,t3,t4,t5)
    for i in range(5):
        linhas[i].set_data(pontos[i:i+2,0], pontos[i:i+2,1])
        linhas[i].set_3d_properties(pontos[i:i+2,2])
    
    # Atualiza gráfico X e Z da ponta em função de θ5
    x_vals.clear()
    z_vals.clear()
    for angle in theta5_vals:
        p = calcular_pontos(t1,t2,t3,t4,angle)
        x_vals.append(p[-1,0])
        z_vals.append(p[-1,2])
    linhaX.set_data(theta5_vals, x_vals)
    linhaZ.set_data(theta5_vals, z_vals)
    fig.canvas.draw_idle()
    fig2.canvas.draw_idle()

slider1.on_changed(update)
slider2.on_changed(update)
slider3.on_changed(update)
slider4.on_changed(update)
slider5.on_changed(update)

update(0)  # Inicializa os gráficos
plt.show()
