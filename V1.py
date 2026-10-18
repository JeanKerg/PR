import openturns as ot
import numpy as np

# 1. Paramètres 
n = 1000        # Taille de l'échantillon initial (X_n,Y_n)
m = 500         # Taille de l'échantillon supplémentaire
k = 100         # Nombre de valeurs extrêmes pour n
k_plus = int(k * (n + m) / n)   # Nombre d'extrêmes pour n+m tel que beta = 1
N_sim = 10000   # Nombre de simulations
distribution = "frechet"         # Distribution
# Cauchy 
s = 0        # Paramètre de dépendance
# Frechet
theta = 0.1

# 2. Distribution
if distribution == "frechet":
    margin1 = ot.Frechet(1, 1, 0)       # Marginale de loi de Fréchet standard, avec gamma = 1 (1er paramètre)
    margin2 = ot.Frechet(1, 1, 0)   
    copula = ot.GumbelCopula(1/theta)   # Copule de Gumbel équivalent à la dépendance logistique
    distribution = ot.JointDistribution([margin1, margin2], copula) # On joint les deux lois avec la copule
    margin_m = margin2                  # Echantillon m tiré indépendamment

elif distribution == "cauchy":
    # La loi de Cauchy est une loi de Student avec 1 degré de liberté
    S = ot.CorrelationMatrix(2) # Matrice d'échelle S (diagonale = 1, hors-diagonale = s)
    S[0, 1] = s
    S[1, 0] = s
    
    # Student(degré de liberté, centre, dispersion, dépendance)
    distribution = ot.Student(1, [0, 0], [1, 1], S)
    
    # La marginale 2 est une Cauchy univariée
    margin_m = ot.Student(1, [0], [1], ot.CorrelationMatrix(1))

else:
    raise ValueError("Le mode doit être 'cauchy' ou 'frechet'.")

# 3. Fonction de l'estimateur de Hill
def hill_estimator(data, k_val):
    """Calcule l'estimateur de Hill sur les k_val plus grandes valeurs."""
    sorted_data = np.sort(data)
    threshold = sorted_data[-k_val - 1] # Seuil X(n-k)
    extremes = sorted_data[-k_val:]     # Les k valeurs au-dessus du seuil
    return np.mean(np.log(extremes) - np.log(threshold))

# 4. Simulation
gamma1_results = np.zeros(N_sim)    # Listes pour stocker les résultats des 10 000 simulations
gamma_adapted_results = np.zeros(N_sim)

for i in range(N_sim):
    # A. Génération des données
    # Échantillon conjoint
    sample_n = np.array(distribution.getSample(n))  # Génère les n données
    X_n = sample_n[:, 0]                           # Les X1, ..., Xn
    Y_n = sample_n[:, 1]                           # Les Y1, ..., Yn
    
    # Échantillon des m données supplémentaires, indépendantes pour Y
    sample_m = np.array(margin_m.getSample(m)).flatten()
    
    # Fusion pour avoir l'échantillon total Y1, ..., Ym
    Y_n_m = np.concatenate((Y_n, sample_m))
    
    # B. Calcul des estimateurs de base
    gamma1 = hill_estimator(X_n, k)
    gamma2 = hill_estimator(Y_n, k)
    gamma2_plus = hill_estimator(Y_n_m, k_plus)
    
    # C. Calcul empirique de la copule de queue R(1,1)
    # On compte combien de fois X_n et Y_n dépassent simultanément leur seuil respectif (8)
    threshold_X_n = np.sort(X_n)[-k - 1]
    threshold_Y_n = np.sort(Y_n)[-k - 1]
    
    joint_extremes_count = np.sum((X_n > threshold_X_n) & (Y_n > threshold_Y_n))
    R_11_hat = joint_extremes_count / k
    
    # D. Le nouvel estimateur adapté (Simplification de (10) car beta = 1)
    gamma_adapted = gamma1 + (gamma1 / gamma2_plus) * R_11_hat * (gamma2_plus - gamma2)
    
    # E. Stockage
    gamma1_results[i] = gamma1
    gamma_adapted_results[i] = gamma_adapted

# 5. Analyse des résultats
var_classique = np.var(gamma1_results)
var_adapte = np.var(gamma_adapted_results)

mse_classique = np.mean((gamma1_results - 1)**2)    # gamma = 1 par hypothèse
mse_adapte = np.mean((gamma_adapted_results - 1)**2)

reduction_variance = (1 - (var_adapte / var_classique)) * 100 # Calcul de pourcentage classique
reduction_mse = (1 - (mse_adapte / mse_classique)) * 100

print(f"Variance Hill classique   : {var_classique:.5f}")
print(f"Variance Hill adapté      : {var_adapte:.5f}")
print(f"--> Réduction de variance : {reduction_variance:.1f} %")
print(f"--> Réduction du MSE      : {reduction_mse:.1f} %")
print(f"--> R(1,1)                : {R_11_hat:.2f}")