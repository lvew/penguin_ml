import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.title("Clasificador de pingüinos: una aplicación de aprendizaje de máquina")
st.write(
    "Esta app usa seis inputs para predecir la especie de pingüinos, usando "
    "un modelo construido en los datos de los pingüinos de Palmer. ¡Usa las opciones de abajo"
    " para iniciar!"
)

password_guess = st.text_input("¿Cuál es la contraseña?")
if password_guess != st.secrets["password"]:
    st.stop()

penguin_file = st.file_uploader("Sube tus propios datos de pingüinos")

penguin_df = pd.read_csv("penguins.csv")
rf_pickle = open("random_forest_penguin.pickle", "rb")
map_pickle = open("output_penguin.pickle", "rb")
rfc = pickle.load(rf_pickle)
unique_penguin_mapping = pickle.load(map_pickle)
rf_pickle.close()
map_pickle.close()

with st.form("user_input"):
    island = st.selectbox("Isla del pingüino", options=["Biscoe", "Dream", "Torgerson"])
    sex = st.selectbox("Sexo", options=["Hembra", "Macho"])
    bill_length = st.number_input("Longitud del pico (mm)", min_value=0)
    bill_depth = st.number_input("Profundidad del pico (mm)", min_value=0)
    flipper_length = st.number_input("Longitud de la aleta (mm)", min_value=0)
    body_mass = st.number_input("Masa corporal (g)", min_value=0)
    st.form_submit_button("Enviar")

island_biscoe, island_dream, island_torgerson = 0, 0, 0
if island == "Biscoe":
    island_biscoe = 1
elif island == "Dream":
    island_dream = 1
elif island == "Torgerson":
    island_torgerson = 1

sex_female, sex_male = 0, 0
if sex == "Hembra":
    sex_female = 1
elif sex == "Macho":
    sex_male = 1

new_prediction = rfc.predict(
    [
        [
            bill_length,
            bill_depth,
            flipper_length,
            body_mass,
            island_biscoe,
            island_dream,
            island_torgerson,
            sex_female,
            sex_male,
        ]
    ]
)
prediction_species = unique_penguin_mapping[new_prediction][0]

st.subheader("Prediciendo la especie de tu pingüino:")
st.write("Predecimos que tu pingüino es de la especie {} ".format(prediction_species))
st.write(
    "Usamos un modelo de aprendizaje de máquina (Random Forest) para "
    "predecir la especie. Las características que se usan en esta predicción "
    "se ordenan por importancia relativa en la siguiente gráfica."
)
st.image("feature_importance.png")

st.write(
    "A continuación se encuentran los histogramas para cada variable continua"
    " según la especie del pingüino. La línea vertical "
    "representa el valor que tu colocaste."
)

fig, ax = plt.subplots()
ax = sns.displot(x=penguin_df["bill_length_mm"], hue=penguin_df["species"])
plt.axvline(bill_length)
plt.title("Longitud del pico por especie")
st.pyplot(ax)

fig, ax = plt.subplots()
ax = sns.displot(x=penguin_df["bill_depth_mm"], hue=penguin_df["species"])
plt.axvline(bill_depth)
plt.title("Profundidad del pico poe especie")
st.pyplot(ax)

fig, ax = plt.subplots()
ax = sns.displot(x=penguin_df["flipper_length_mm"], hue=penguin_df["species"])
plt.axvline(flipper_length)
plt.title("Longitud de la aleta por especie")
st.pyplot(ax)
