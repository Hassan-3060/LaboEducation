import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x_experience = 20 * np.random.rand(100, 1)

bruit = 5000* np.random.randn(100, 1)
y_salaire = 5000 + 2000 * x_experience + bruit

print("🤖 L'IA analyse les 100 employés...")

model = LinearRegression()
model.fit(x_experience, y_salaire)

print(f"💰 Augmentation annuelle devinée par l'IA : {model.coef_[0][0]:.2f}€ (Réalité : 2000€)")
print(f"🏁 Salaire de départ deviné par l'IA      : {model.intercept_[0]:.2f}€ (Réalité : 30000€)")

prediction_10ans = model.predict([[10]])
print(f"\n🔮 Prédiction pour 10 ans d'expérience : {prediction_10ans[0][0]:.2f}€")

# 5. VISUALISATION DU COMPROMIS
plt.figure(figsize=(10, 6))
# Les points bleus seront éparpillés (le chaos)
plt.scatter(x_experience, y_salaire, color='blue', alpha=0.5, label='Employés Réels (Données + Bruit)')
# La ligne rouge coupe au milieu (le compromis)
plt.plot(x_experience, model.predict(x_experience), color='red', linewidth=3, label='Logique détectée par l\'IA')

plt.title('Salaire vs Expérience (L\'IA cherche l\'ordre dans le chaos)')
plt.xlabel('Années d\'expérience')
plt.ylabel('Salaire Annuel (€)')
plt.legend()
plt.grid(True)
plt.savefig('graphique_salaire.png')
print("\n🖼️  Image 'graphique_salaire.png' générée !")