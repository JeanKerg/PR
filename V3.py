import openturns as ot
import numpy as np

# 1. Paramètres 
n = 1000       # Taille de l'échantillon initial (X_n,Y_n)
m = 1000       # Taille de l'échantillon supplémentaire
k = 100        # Nombre de valeurs extrêmes pour n
k_plus = int(k * (n + m) / n)   # Nombre d'extrêmes pour n+m tel que beta = 1
N_sim = 10000  # Nombre de simulations
distribution_init = "cauchy"         # Distribution
d = 3           # Dimension
# Cauchy 
s = 0           # Paramètre de dépendance
r = 0
# Frechet
theta = 0.3

# 2. Distribution
if distribution_init == "frechet":
    margin1 = ot.Frechet(1, 1, 0)       # Marginale de loi de Fréchet standard, avec gamma = 1 (1er paramètre)
    margin2 = ot.Frechet(1, 1, 0)   
    copula = ot.GumbelCopula(1/theta)   # Copule de Gumbel équivalent à la dépendance logistique
    distribution = ot.JointDistribution([margin1, margin2], copula) # On joint les deux lois avec la copule
    margin_m = margin2                  # Echantillon m tiré indépendamment

elif distribution_init == "cauchy":
    # La loi de Cauchy est une loi de Student avec 1 degré de liberté
    if d == 2 : 
        S = ot.CorrelationMatrix(2, [
            1, s, 
            s, 1])
    if d == 3 : 
        S = ot.CorrelationMatrix(3, [
            1, s, s,
            s, 1, r,
            s, r, 1])
    # Student(degré de liberté, centre, dispersion, dépendance)
    distribution = ot.Student(1, np.zeros(d), np.ones(d), S)    

else:
    raise ValueError("Le mode doit être 'cauchy' ou 'frechet'.")

# 3. Fonction de l'estimateur de Hill
def hill_estimator(data, k_val):
    # Calcule l'estimateur de Hill sur les k_val plus grandes valeurs.
    sorted_data = np.sort(data)
    threshold = sorted_data[-k_val - 1] # Seuil X(n-k)
    extremes = sorted_data[-k_val:]     # Les k valeurs au-dessus du seuil
    return np.mean(np.log(extremes) - np.log(threshold))

# 4. Simulation
gamma1_results = np.zeros(N_sim)    # Listes pour stocker les résultats des 10 000 simulations
gamma_adapted_results = np.zeros(N_sim)
R12_results = np.zeros(N_sim)
R13_results = np.zeros(N_sim)
R23_results = np.zeros(N_sim)

for i in range(N_sim):
    ot.RandomGenerator.SetSeed(i); np.random.seed(i)
    # A. Génération des données
    # Échantillon conjoint
    sample_n = np.array(distribution.getSample(n))          # Génère les n données

    if distribution_init == "cauchy" : 
        # On se restreint au premier quadrant
        while np.any(sample_n <= 0):    # Tous les X et Y doivent être > 0
            liste_bool = np.any(sample_n <= 0, axis=1)  # Tableau 1D avec True / False
            sample_n[liste_bool] = np.array(distribution.getSample(int(np.sum(liste_bool)))) # Change uniquement aux indices ayant True, pour le bon nombre de corrections
    
        # On Génère les m Y et Z supplémentaires dans le 1er quadrant
        sample_m1 = np.array(distribution.getSample(m))
        while np.any(sample_m1 <= 0):
            liste_bool = np.any(sample_m1 <= 0, axis = 1)
            sample_m1[liste_bool] = np.array(distribution.getSample(int(np.sum(liste_bool))))

        # On conserve uniquement Y et Z
        sample_m = sample_m1[:, 1:d]

    else : 
        sample_m = np.array(margin_m.getSample(m)).flatten()    # Échantillon des m données supplémentaires, indépendantes pour Y
    
    X_n = sample_n[:, 0]                           # Les X1, ..., Xn
    Y_n = sample_n[:, 1]                           # Les Y1, ..., Yn
    if d == 3 : 
        Z_n = sample_n[:, 2]

    # Fusion pour avoir l'échantillon total Y1, ..., Ym
    Y_n_m = np.concatenate((Y_n, sample_m[:, 0]))
    if d == 3 : 
        Z_n_m = np.concatenate((Z_n, sample_m[:, 1]))
    
    # B. Calcul des estimateurs de base
    gamma1 = hill_estimator(X_n, k)
    gamma2 = hill_estimator(Y_n, k)
    if d == 3 : 
        gamma3 = hill_estimator(Z_n, k)
    gamma2_plus = hill_estimator(Y_n_m, k_plus)
    if d == 3 :
        gamma3_plus = hill_estimator(Z_n_m, k_plus)
    
    # C. Calcul empirique des copule de queue R(1,1) chapeau
    seuils = np.sort(sample_n, axis=0)[-k - 1, :]       # On trie chaque colonne de sample_n et on détermine les seuils pour les k plus grandes valeurs
    depasse = sample_n > seuils                         # On crée un tableau True/False
    R12_hat = np.sum(depasse[:, 0] & depasse[:, 1]) / k # On applique la formule 8
    if d == 3 : 
        R13_hat = np.sum(depasse[:, 0] & depasse[:, 2]) / k
        R23_hat = np.sum(depasse[:, 1] & depasse[:, 2]) / k
    
    # D. Le nouvel estimateur adapté (p561 et 562)
    if d == 2 : 
        gamma_adapted = gamma1 + (gamma1 / gamma2_plus) * R12_hat * (gamma2_plus - gamma2)
    else : 
        denom = 1 - R23_hat**2  # Le dénominateur ne doit pas être nul
        
        if denom == 0 : 
            raise ValueError("Dénominateur nul : R23_hat = 1.")
        
        coeff2 = (R12_hat - R13_hat * R23_hat) / denom
        coeff3 = (R13_hat - R12_hat * R23_hat) / denom

        gamma_adapted = (gamma1 + (gamma1 / gamma2_plus) * coeff2 * (gamma2_plus - gamma2) + (gamma1 / gamma3_plus) * coeff3 * (gamma3_plus - gamma3))
    
    # E. Stockage
    gamma1_results[i] = gamma1
    gamma_adapted_results[i] = gamma_adapted
    R12_results[i] = R12_hat
    if d == 3 : 
        R13_results[i] = R13_hat
        R23_results[i] = R23_hat

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
print(f"--> R12(1,1) moyen        : {np.mean(R12_results):.2f}")
if d == 3 : 
    print(f"--> R13(1,1) moyen        : {np.mean(R13_results):.2f}")
    print(f"--> R23(1,1) moyen        : {np.mean(R23_results):.2f}")