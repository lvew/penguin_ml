import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle

st.title("Clasificador de pingüinos: una aplicación de aprendizaje de máquina")
st.write(
    "Esta app usa seis inputs para predecir la especie de pingüinos, usando "
    "un modelo construido en los datos de los pingüinos de Palmer. ¡Usa las opciones de abajo"
    " para iniciar!"
)

penguin_df = pd.read_csv("penguins.csv")
rf_pickle = open("random_forest_penguin.pickle", "rb")
map_pickle = open("output_penguin.pickle", "rb")
rfc = pickle.load(rf_pickle)
unique_penguin_mapping = pickle.load(map_pickle)
rf_pickle.close()
map_pickle.close()

with st.form("user_input"):
    island = st.selectbox("Penguin Island", options=["Biscoe", "Dream", "Torgerson"])
    sex = st.selectbox("Sex", options=["Female", "Male"])
    bill_length = st.number_input("Bill Length (mm)", min_value=0)
    bill_depth = st.number_input("Bill Depth (mm)", min_value=0)
    flipper_length = st.number_input("Flipper Length (mm)", min_value=0)
    body_mass = st.number_input("Body Mass (g)", min_value=0)
    st.form_submit_button()

island_biscoe, island_dream, island_torgerson = 0, 0, 0
if island == "Biscoe":
    island_biscoe = 1
elif island == "Dream":
    island_dream = 1
elif island == "Torgerson":
    island_torgerson = 1

sex_female, sex_male = 0, 0
if sex == "Female":
    sex_female = 1
elif sex == "Male":
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
    "predecir la especie, las características que se usan en esta predicción "
    "se ordenan por importancia relativa abajo."
)
st.image("feature_importance.png")

st.write(
    "Abajo se encuentran los histogramas para cada variable continua"
    " seperados por la especie del pingüino. La línea vertical "
    "representa el valor que tu colocaste."
)

fig, ax = plt.subplots()
ax = sns.displot(x=penguin_df["bill_length_mm"], hue=penguin_df["species"])
plt.axvline(bill_length)
plt.title("Bill Length by Species")
st.pyplot(ax)

fig, ax = plt.subplots()
ax = sns.displot(x=penguin_df["bill_depth_mm"], hue=penguin_df["species"])
plt.axvline(bill_depth)
plt.title("Bill Depth by Species")
st.pyplot(ax)

fig, ax = plt.subplots()
ax = sns.displot(x=penguin_df["flipper_length_mm"], hue=penguin_df["species"])
plt.axvline(flipper_length)
plt.title("Flipper Length by Species")
st.pyplot(ax)
