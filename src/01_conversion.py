import numpy as np
from sklearn.linear_model import LinearRegression  # <-- Correction de l'import

# 1. LES DONNÉES
# Il faut impérativement le même nombre de lignes pour X et y
question_training = np.array([[-40], [-20], [0], [20], [40], [60], [80], [100], [120], [140]])

# J'ai calculé les bonnes réponses pour que ça matche (10 valeurs)
answer_training = np.array([
    [233.15], [253.15], [273.15], [293.15], [313.15], 
    [333.15], [353.15], [373.15], [393.15], [413.15]
])

model = LinearRegression()

print("Le modèle commence l'entraînement...")
model.fit(question_training, answer_training)
print("Le modèle a terminé l'entraînement.")

# Le input doit être converti en float car input() renvoie du texte
temp_celsius = float(input("Entrez la température en degrés Celsius : "))
predication = model.predict([[temp_celsius]])

print(f"La température en Kelvin est de : {predication[0][0]:.2f} K")

print("\n--- 🔍 Inspection du cerveau de l'IA ---")
# Notez bien : Pour Kelvin, le multiplicateur est 1, pas 1.8 !
print(f"Poids (Le multiplicateur) trouvé : {model.coef_[0][0]:.4f} (Devrait être ~1.0)")
print(f"Biais (L'addition) trouvé        : {model.intercept_[0]:.4f} (Devrait être ~273.15)")


# Voir le cerveau de l'IA
import matplotlib.pyplot as plt
print("\n🎨 Génération du graphique en cours...")

plt.scatter(question_training, answer_training, color='blue', label='Données Réelles')
plt.plot(question_training, model.predict(question_training), color='red', linewidth=2, label='Prédiction IA')

plt.title('La découverte de la loi Kelvin par l\'IA')
plt.xlabel('Celsius')
plt.ylabel('Kelvin')
plt.legend()
plt.grid(True)

plt.savefig('mon_graphique_ia.png')
print("✅ Image sauvegardée sous 'mon_graphique_ia.png' ! Regardez dans l'explorateur à gauche.")