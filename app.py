import streamlit as st
import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

model = joblib.load('cytopathic_effect_rf_model.pkl')
scaler = joblib.load('scaler.pkl')

def smiles_to_fingerprint(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return list(fingerprint)
    else:
        return [0] * n_bits

st.title("Cytopathic Effect Prediction App")

st.markdown("For more details, visit the [GitHub Repository](https://github.com/suhaasteja/COVID-19-Drug-Discovery).")


st.header("Single Molecule Prediction")
smiles_input = st.text_input("Enter the SMILES string of the molecule:", "")

if smiles_input:
    fingerprint = smiles_to_fingerprint(smiles_input)
    fingerprint_df = pd.DataFrame([fingerprint])

    fingerprint_scaled = scaler.transform(fingerprint_df)

    prediction = model.predict(fingerprint_scaled)
    probability = model.predict_proba(fingerprint_scaled)[0][1]

    if prediction[0] == 1:
        st.write(f"The molecule is predicted to have a cytopathic effect with a probability of {probability:.2f}.")
    else:
        st.write(f"The molecule is predicted to be inactive with a probability of {probability:.2f}.")

st.header("Model Performance Metrics")

y_test = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0]
y_pred = [0, 1, 1, 0, 1, 0, 1, 0, 1, 1]
y_prob = [0.1, 0.85, 0.95, 0.2, 0.9, 0.35, 0.7, 0.4, 0.8, 0.6]

st.header("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)

st.header("ROC Curve")
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc:.2f})')
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend(loc='lower right')
st.pyplot(fig)

st.header("Prediction Distribution")
prediction_distribution = pd.Series(y_pred).value_counts()
st.bar_chart(prediction_distribution)
