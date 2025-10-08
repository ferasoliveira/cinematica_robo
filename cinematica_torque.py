# cinematica_5R_euler_lagrange_status_fix_simplify_3Dtraj.py
"""
Versão: 5R com Euler-Lagrange e log detalhado de tempo
Melhorias:
 - Logs com tempo total e tempo entre etapas
 - Cache automático (Jacobianos e D)
 - Removido simplify pesado
 - Adicionada visualização 3D da trajetória final
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import pickle
from mpl_toolkits.mplot3d import Axes3D

# ---------------------------
# Funções auxiliares de tempo e log
# ---------------------------
t_inicio = time.time()
t_prev = t_inicio

def tempo_formatado():
    total = time.time() - t_inicio
    m, s = divmod(total, 60)
    return f"[{int(m):02d}:{int(s):02d}]"

def log_etapa(msg):
    global t_prev
    agora = time.time()
    delta = agora - t_prev
    t_prev = agora
    print(f"{tempo_formatado()} log {{etapa = {delta:.2f}s}} {msg}")

# Início
log_etapa("✅ Iniciando script de modelagem 5R com Euler-Lagrange")

# ---------------------------
# 1) Variáveis simbólicas e DH
# ---------------------------
log_etapa("🔧 Definindo variáveis simbólicas e parâmetros DH")

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
log_etapa("✅ Matrizes DH e transformações homogêneas configuradas")

# ---------------------------
# 2) Jacobiano simbólico
# ---------------------------
log_etapa("🧩 Calculando Jacobiano simbólico")

o0 = sp.Matrix([0,0,0]); z0 = sp.Matrix([0,0,1])
origins = [o0, T1[:3,3], T2[:3,3], T3[:3,3], T4[:3,3]]
zs = [z0, T1[:3,2], T2[:3,2], T3[:3,2], T4[:3,2]]
o5 = T5[:3,3]

Jv_cols = [ zs[i].cross(o5 - origins[i]) for i in range(5) ]
Jw_cols = [ zs[i] for i in range(5) ]
J_sym = sp.Matrix.vstack(sp.Matrix.hstack(*Jv_cols), sp.Matrix.hstack(*Jw_cols))
J_func = sp.lambdify((theta1,theta2,theta3,theta4,theta5), J_sym.subs(valores_fixos), 'numpy')
log_etapa("✅ Jacobiano simbólico concluído")

# ---------------------------
# 3) Função para pontos numéricos
# ---------------------------
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
# 4) Simulação da trajetória
# ---------------------------
log_etapa("🚀 Iniciando simulação da trajetória")
np.random.seed()
q0 = (np.random.rand(5) - 0.5) * np.pi
qdot = (np.random.rand(5) - 0.5) * 0.4  

dt, steps = 0.05, 200
q = np.array(q0, dtype=float)
traj_points, endeff_traj, vel_traj = [], [], []

for _ in range(steps):
    pts = calcular_pontos_num(q)
    traj_points.append(pts)
    endeff_traj.append(pts[-1])
    Jn = np.array(J_func(*q), dtype=float)
    xdot = Jn @ qdot
    vel_traj.append(xdot[:3])
    q = q + qdot * dt

traj_points = np.array(traj_points)
endeff_traj = np.array(endeff_traj)
vel_traj = np.array(vel_traj)
log_etapa("✅ Simulação da trajetória concluída")

# ---------------------------
# 5) Modelagem dinâmica por Euler-Lagrange
# ---------------------------
log_etapa("⚙️ Iniciando modelagem dinâmica (Euler-Lagrange)")

qsyms = sp.Matrix([theta1,theta2,theta3,theta4,theta5])
qd_syms = sp.Matrix(sp.symbols('q1d q2d q3d q4d q5d'))
qdd_syms = sp.Matrix(sp.symbols('q1dd q2dd q3dd q4dd q5dd'))

m1,m2,m3,m4,m5 = sp.symbols('m1 m2 m3 m4 m5', positive=True)
l1,l2,l3,l4,l5 = a2, a3, a4, a5, sp.symbols('l5')
I1_sym, I2_sym, I3_sym, I4_sym, I5_sym = sp.symbols('I1 I2 I3 I4 I5', positive=True)

masses = [m1,m2,m3,m4,m5]
Is = [I1_sym, I2_sym, I3_sym, I4_sym, I5_sym]
lengths = [l1,l2,l3,l4,l5]

origins_sym = [sp.Matrix([0,0,0]), T1[:3,3], T2[:3,3], T3[:3,3], T4[:3,3], T5[:3,3]]

cache_file = "jacobianos_cache.pkl"
if os.path.exists(cache_file):
    try:
        log_etapa("📦 Cache encontrado, carregando Jacobianos e matriz D...")
        with open(cache_file, "rb") as f:
            Jv_c, Jw_c, D = pickle.load(f)
        log_etapa("✅ Cache carregado com sucesso")
    except Exception as e:
        log_etapa(f"⚠️ Falha ao carregar cache (vai recalcular): {e}")
        os.remove(cache_file)
        Jv_c, Jw_c, D = None, None, None
else:
    Jv_c, Jw_c, D = None, None, None

if Jv_c is None or Jw_c is None or D is None:
    log_etapa("📐 Calculando posições e Jacobianos dos centros de massa (1/2)")
    oc_sym = []
    for i in range(5):
        # Não chamamos simplify pesado aqui; calculamos o ponto médio direto
        oc = origins_sym[i] + sp.Rational(1,2)*(origins_sym[i+1] - origins_sym[i])
        oc_sym.append(sp.Matrix(oc))
    log_etapa("✅ Posições dos centros de massa calculadas")

    Jv_c, Jw_c = [], []
    for i in range(5):
        ti = time.time()
        # calculo dos Jv_c[i] e Jw_c[i] sem simplificação pesada
        Jv_cols_ci = [ zs[j].cross(oc_sym[i] - origins_sym[j]) for j in range(5) ]
        Jv_ci = sp.Matrix.hstack(*Jv_cols_ci)
        Jv_c.append(Jv_ci)  # sem sp.simplify para evitar expansão custosa
        Jw_cols_ci = [ zs[j] for j in range(5) ]
        Jw_c.append(sp.Matrix.hstack(*Jw_cols_ci))
        log_etapa(f"✅ Jacobianos do elo {i+1} calculados (tempo interno = {time.time()-ti:.2f}s)")

    log_etapa("📊 Montando matriz de inércia D(q) (1/2)")
    D = sp.zeros(5,5)
    for i in range(5):
        ti = time.time()
        D += masses[i] * (Jv_c[i].T * Jv_c[i])
        D += Is[i] * (Jw_c[i].T * Jw_c[i])
        log_etapa(f"✅ Incremento de D com elo {i+1} concluído (tempo interno = {time.time()-ti:.2f}s)")

    # **NÃO** chamar sp.simplify(D) aqui — costuma travar/explodir.
    log_etapa("⚠️ Pulando sp.simplify(D) intencionalmente (muito custoso).")

    # Salvar cache
    try:
        with open(cache_file, "wb") as f:
            pickle.dump((Jv_c, Jw_c, D), f)
        log_etapa("💾 Cache (Jv_c, Jw_c, D) salvo com sucesso")
    except Exception as e:
        log_etapa(f"⚠️ Falha ao salvar cache: {e}")

# ---------------------------
# 6) Matriz de Coriolis C(q,qdot)
# ---------------------------
log_etapa("🔁 Calculando matriz de Coriolis C(q,qdot)... (isso pode demorar mas evita simplify(D))")

n = 5
# c[i][j][k] = 1/2 * (dD[k,j]/dq_i + dD[k,i]/dq_j - dD[i,j]/dq_k)
c = [[[None]*n for _ in range(n)] for __ in range(n)]
for i in range(n):
    for j in range(n):
        for k in range(n):
            # Não usamos sp.simplify aqui para reduzir custo geral
            c[i][j][k] = sp.Rational(1,2)*(sp.diff(D[k,j], qsyms[i]) + sp.diff(D[k,i], qsyms[j]) - sp.diff(D[i,j], qsyms[k]))

C = sp.zeros(n, n)
for j in range(n):
    for k in range(n):
        s = 0
        for i in range(n):
            s += c[i][j][k]*qd_syms[i]
        C[k,j] = s

log_etapa("✅ Matriz de Coriolis C(q,qdot) calculada (sem simplificações excessivas)")

# ---------------------------
# 7) Vetor gravitacional G(q)
# ---------------------------
log_etapa("🌍 Calculando vetor gravitacional G(q)")
g = sp.symbols('g')
# P usando os centros de massa calculados em oc_sym (ou origins_sym como fallback)
# Note: oc_sym já foi criado acima
P = sum(masses[i] * g * sp.simplify(oc_sym[i][2]) for i in range(5))
G = sp.Matrix([sp.diff(P, qsyms[i]) for i in range(n)])
# pequeno simplify local aceitável
G = sp.Matrix([sp.simplify(G[i]) for i in range(n)])
log_etapa("✅ Vetor de gravidade G(q) concluído")

# ---------------------------
# 8) Montagem τ = D qdd + C qd + G
# ---------------------------
log_etapa("🧮 Montando equação simbólica de torques τ = D q̈ + C q̇ + G")
tau_sym = D * qdd_syms + C * qd_syms + G
# Não simplificamos globalmente (evita expansão gigante)
log_etapa("✅ Expressão simbólica de torque montada (sem simplify final)")

# ---------------------------
# 9) Criar função numérica para torque
# ---------------------------
log_etapa("🧰 Criando função numérica para cálculo de torque (lambdify)")

# subs dos Is para expressões de inércia em função das massas e comprimentos
subs_inertia = {}
for i in range(5):
    Li = lengths[i]
    Mi = masses[i]
    subs_inertia[Is[i]] = sp.simplify(Mi * Li**2 / 12)

subs_fix = {d1:valores_fixos[d1], a2:valores_fixos[a2], a3:valores_fixos[a3], a4:valores_fixos[a4], a5:valores_fixos[a5]}
subs_fix[sp.symbols('l5')] = valores_fixos[a5]

all_syms = [theta1,theta2,theta3,theta4,theta5] + list(qd_syms) + list(qdd_syms) + [m1,m2,m3,m4,m5,g]

# aplicamos substituições de inércia e parâmetros fixos para reduzir símbolos antes do lambdify
tau_num_expr = tau_sym.subs(subs_inertia).subs(subs_fix)

tau_func = sp.lambdify(all_syms, tau_num_expr, 'numpy')
log_etapa("✅ Função numérica de torque criada (via lambdify)")

# ---------------------------
# 10) Avaliação numérica
# ---------------------------
log_etapa("📈 Avaliando torques ao longo da trajetória numericamente")

num_params = {m1: 2.0, m2: 2.0, m3: 1.5, m4: 1.0, m5: 0.8, g: 9.81}

q_traj = []
q = np.array(q0, dtype=float)
for _ in range(steps):
    q_traj.append(q.copy())
    q = q + qdot * dt
q_traj = np.array(q_traj)

qd_vals = tuple(qdot.tolist())
qdd_vals = tuple([0.0]*5)

taus = []
for i in range(len(q_traj)):
    qvals = tuple(q_traj[i].tolist())
    argvals = qvals + qd_vals + qdd_vals + (num_params[m1],num_params[m2],num_params[m3],num_params[m4],num_params[m5], num_params[g])
    tau_eval = np.array(tau_func(*argvals), dtype=float).flatten()
    taus.append(tau_eval)
taus = np.array(taus)
log_etapa("✅ Torques avaliados numericamente")

# ---------------------------
# 11) Gráfico dos torques
# ---------------------------
log_etapa("📊 Gerando gráfico dos torques")
t = np.arange(0, len(taus))*dt
plt.figure(figsize=(10,6))
for j in range(5):
    plt.plot(t, taus[:,j], label=f'τ{j+1}')
plt.xlabel("Tempo (s)")
plt.ylabel("Torque (N·m)")
plt.title("Torques nas juntas (Euler-Lagrange)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("torques_juntas.png", dpi=150)
plt.show()
log_etapa("✅ Gráfico de torques salvo em 'torques_juntas.png'")

# ---------------------------
# 12) Trajetórias XY, XZ, YZ
# ---------------------------
log_etapa("🖼️ Salvando trajetórias XY, XZ e YZ")

def salvar_trajetoria(x, y, xlabel, ylabel, nome):
    plt.figure()
    plt.plot(x, y, '-')
    plt.scatter(x[0], y[0], s=80, color='green', label="Partida")
    plt.scatter(x[-1], y[-1], s=80, color='red', label="Chegada")
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.title(f"Trajetória {xlabel}{ylabel}")
    plt.legend()
    plt.grid(True)
    plt.savefig(nome, dpi=150)
    plt.close()

salvar_trajetoria(endeff_traj[:,0], endeff_traj[:,1], "X (m)", "Y (m)", "traj_XY.png")
salvar_trajetoria(endeff_traj[:,0], endeff_traj[:,2], "X (m)", "Z (m)", "traj_XZ.png")
salvar_trajetoria(endeff_traj[:,1], endeff_traj[:,2], "Y (m)", "Z (m)", "traj_YZ.png")
log_etapa("✅ Trajetórias XY, XZ e YZ salvas com sucesso")

# ---------------------------
# 13) Trajetória 3D final
# ---------------------------
log_etapa("🌐 Gerando trajetória 3D do efetuador final")

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(endeff_traj[:,0], endeff_traj[:,1], endeff_traj[:,2], label='Trajetória', lw=2)
ax.scatter(endeff_traj[0,0], endeff_traj[0,1], endeff_traj[0,2], color='green', s=80, label='Partida')
ax.scatter(endeff_traj[-1,0], endeff_traj[-1,1], endeff_traj[-1,2], color='red', s=80, label='Chegada')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title("Trajetória 3D do efetuador final")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig("traj_3D.png", dpi=150)
plt.show()
log_etapa("✅ Trajetória 3D salva como 'traj_3D.png'")

log_etapa("🎯 Execução concluída com sucesso")