# cinematica_diferencial.py
"""
Cinemática diferencial para robô 5R.
- Jacobiano geométrico simbólico.
- Simulação diferencial xdot = J(q)*qdot.
- Animação 3D do braço.
- Salva as trajetórias XY, XZ e YZ como imagens separadas.
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ---------------------------
# 1) Variáveis simbólicas e DH
# ---------------------------
theta1, theta2, theta3, theta4, theta5 = sp.symbols('theta1 theta2 theta3 theta4 theta5')
d1, a2, a3, a4, a5 = sp.symbols('d1 a2 a3 a4 a5')

def dh_matrix(theta, d, a, alpha):
    return sp.Matrix([
        [sp.cos(theta), -sp.sin(theta)*sp.cos(alpha),  sp.sin(theta)*sp.sin(alpha), a*sp.cos(theta)],
        [sp.sin(theta),  sp.cos(theta)*sp.cos(alpha), -sp.cos(theta)*sp.sin(alpha), a*sp.sin(theta)],
        [0,              sp.sin(alpha),               sp.cos(alpha),               d],
        [0, 0, 0, 1]
    ])

A1 = dh_matrix(theta1, d1, 0, sp.pi/2)
A2 = dh_matrix(theta2, 0, a2, 0)
A3 = dh_matrix(theta3, 0, a3, 0)
A4 = dh_matrix(theta4, 0, a4, 0)
A5 = dh_matrix(theta5, 0, a5, -sp.pi/2)

valores_fixos = {d1: 1.0, a2: 1.0, a3: 1.0, a4: 1.0, a5: 1.0}

T1, T2, T3, T4, T5 = A1, A1*A2, A1*A2*A3, A1*A2*A3*A4, A1*A2*A3*A4*A5

# ---------------------------
# 2) Jacobiano simbólico
# ---------------------------
o0 = sp.Matrix([0,0,0]); z0 = sp.Matrix([0,0,1])
origins = [o0, T1[:3,3], T2[:3,3], T3[:3,3], T4[:3,3]]
zs = [z0, T1[:3,2], T2[:3,2], T3[:3,2], T4[:3,2]]
o5 = T5[:3,3]

Jv_cols = [ zs[i].cross(o5 - origins[i]) for i in range(5) ]
Jw_cols = [ zs[i] for i in range(5) ]
J_sym = sp.Matrix.vstack(sp.Matrix.hstack(*Jv_cols), sp.Matrix.hstack(*Jw_cols))
J_func = sp.lambdify((theta1,theta2,theta3,theta4,theta5), J_sym.subs(valores_fixos), 'numpy')

def calcular_pontos_num(q):
    subs = {theta1:q[0], theta2:q[1], theta3:q[2], theta4:q[3], theta5:q[4], **valores_fixos}
    Tn = [M.evalf(subs=subs) for M in [T1,T2,T3,T4,T5]]
    pts = np.array([
        [0,0,0],
        [float(Tn[0][0,3]), float(Tn[0][1,3]), float(Tn[0][2,3])],
        [float(Tn[1][0,3]), float(Tn[1][1,3]), float(Tn[1][2,3])],
        [float(Tn[2][0,3]), float(Tn[2][1,3]), float(Tn[2][2,3])],
        [float(Tn[3][0,3]), float(Tn[3][1,3]), float(Tn[3][2,3])],
        [float(Tn[4][0,3]), float(Tn[4][1,3]), float(Tn[4][2,3])]
    ])
    return pts

# ---------------------------
# 3) Configurações aleatórias
# ---------------------------
np.random.seed()
q0 = (np.random.rand(5) - 0.5) * np.pi
qdot = (np.random.rand(5) - 0.5) * 0.4  

# ---------------------------
# 4) Simulação diferencial
# ---------------------------
dt, steps = 0.05, 200
q = q0.copy()
traj_points, endeff_traj, vel_traj = [], [], []

for _ in range(steps):
    pts = calcular_pontos_num(q)
    traj_points.append(pts)
    endeff_traj.append(pts[-1])
    Jn = np.array(J_func(*q), dtype=float)
    xdot = Jn @ qdot
    vel_traj.append(xdot[:3])
    q = q + qdot * dt

traj_points, endeff_traj, vel_traj = np.array(traj_points), np.array(endeff_traj), np.array(vel_traj)

# ---------------------------
# 5) Animação 3D
# ---------------------------
fig = plt.figure(figsize=(8,6))
ax3d = fig.add_subplot(111, projection='3d')

ax3d.set_xlim(-4,4); ax3d.set_ylim(-4,4); ax3d.set_zlim(0,4)
ax3d.set_xlabel("X (m)"); ax3d.set_ylabel("Y (m)"); ax3d.set_zlabel("Z (m)")
ax3d.set_title("Braço Robótico 5R - Animação 3D")

lines = [ax3d.plot([],[],[], lw=3)[0] for _ in range(5)]
ee_point, = ax3d.plot([],[],[],'o', markersize=6, color='blue')
ax3d.scatter(endeff_traj[0,0], endeff_traj[0,1], endeff_traj[0,2], c='g', s=80, label="Partida")
ax3d.scatter(endeff_traj[-1,0], endeff_traj[-1,1], endeff_traj[-1,2], c='r', s=80, label="Chegada")
ax3d.legend()

vel_text = ax3d.text2D(0.05, 0.9, "", transform=ax3d.transAxes)

def init():
    for L in lines: L.set_data([], []); L.set_3d_properties([])
    ee_point.set_data([], []); ee_point.set_3d_properties([])
    vel_text.set_text("")
    return lines + [ee_point, vel_text]

def update(frame):
    pts = traj_points[frame]
    for i,L in enumerate(lines):
        L.set_data([pts[i,0],pts[i+1,0]], [pts[i,1],pts[i+1,1]])
        L.set_3d_properties([pts[i,2],pts[i+1,2]])
    ee_point.set_data([pts[-1,0]], [pts[-1,1]])
    ee_point.set_3d_properties([pts[-1,2]])
    v = vel_traj[frame]
    vel_text.set_text(f"Velocidade EE:\nX: {v[0]:.2f} m/s\nY: {v[1]:.2f} m/s\nZ: {v[2]:.2f} m/s")
    return lines + [ee_point, vel_text]

ani = FuncAnimation(fig, update, frames=len(traj_points), init_func=init, interval=50, blit=False)
plt.tight_layout()
plt.show()

# ---------------------------
# 6) Salvar trajetórias XY, XZ, YZ como imagens
# ---------------------------
def salvar_trajetoria(x, y, xlabel, ylabel, nome):
    plt.figure()
    plt.plot(x, y, 'b-')
    plt.scatter(x[0], y[0], c='g', s=80, label="Partida")
    plt.scatter(x[-1], y[-1], c='r', s=80, label="Chegada")
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.title(f"Trajetória {xlabel}{ylabel}")
    plt.legend()
    plt.grid(True)
    plt.savefig(nome, dpi=150)
    plt.close()

salvar_trajetoria(endeff_traj[:,0], endeff_traj[:,1], "X (m)", "Y (m)", "traj_XY.png")
salvar_trajetoria(endeff_traj[:,0], endeff_traj[:,2], "X (m)", "Z (m)", "traj_XZ.png")
salvar_trajetoria(endeff_traj[:,1], endeff_traj[:,2], "Y (m)", "Z (m)", "traj_YZ.png")

print("Trajetórias XY, XZ e YZ salvas como imagens PNG.")
