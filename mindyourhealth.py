import streamlit as st
import pandas as pd
import requests
from azuremodel import azresult
from config import PAGE_CONFIG
import json

st.set_page_config(**PAGE_CONFIG)

def main():

    st.image("data/portada.png", width=880)

    tab0, tab1, tab2, tab3, tab4 = st.tabs(["Intro Health Watcher", "Estudio de Enfermedades con power BI", "Predicción de  Diabetes con Machine Learning", "Drug Analyzer", "Recomendaciones"])

    with tab0:
        st.image("data/drugs.png")
        st.write("Análisis de enfermedades y medicamentos con **:blue[streamlit]**, **:blue[power BI]** y **:blue[sklearn]**.")

        st.markdown("""Los datos de este proyectos proceden de diversas fuentes públicas : 
                          [**CIMA**](https://cima.aemps.es/cima/publico/home.html), [**Drugs**](https://drugs.com), [**INE**](https://datos.gob.es/es/catalogo?publisher_display_name=Instituto+Nacional+de+Estad%C3%ADstica)""")

        st.write("""**`Estudio de Enfermedades con power BI`**  muestra estadísticas sobre enfermedades diagnosticadas en España, estancias y altas hospitalarias.""")

        st.write("""**`Predicción de  Diabetes con Machine Learning`**  calcula su probabilidad de contraer diabetes contestando una pequeña encuesta sobre sus hábitos de vida saludables con una sensibilidad del 90%.""")

        st.write("""**`Drug Analyzer`**  proporciona información sobre medicamentos para tratar diferentes enfermedades.""")

        st.error("""Nota: toda la información proporcionada por esta app es orientativa y basada en datos públicos. Recuerde que ante cualquier duda siempre debe acudir a su médico de cabecera.""")

    with tab1:

        st.components.v1.html('''<iframe title="Diagnosticos" width="880" height="550" src="https://app.powerbi.com/view?r=eyJrIjoiNDVmMGNiZGItZjZhMS00MDQ5LWIwMGEtNjE2ZWIyYjYxMThjIiwidCI6IjYzOTc4MWU1LWJjMmYtNGE3Ni04YmY1LTJiNTg0ZDcxN2U5ZCIsImMiOjl9&filterPaneEnabled=false&navContentPaneEnabled=false" frameborder="0" allowFullScreen="true"></iframe>''', width = 1000, height = 800)
 
    with tab2:

        st.header("ML Predicción de Diabetes según Encuesta")
        st.caption("Recuerda que esta app es sólo informativa, ante cualquier duda médica siempre debes acudir a tu médico de cabecera o consultar a un profesional.")
        st.caption("Modelo de ML RandomForest con un Recall del 89%")
                 
        df = pd.read_csv("data/diabetes_012_health_indicators_BRFSS2015.csv")
        df.drop_duplicates(inplace=True)
        df=df[df['BMI']<50]
        # eliminamos la población con diagnóstico de prediabetes
        df.drop(df[df["Diabetes_012"] == 1].index, axis = 0, inplace= True)
        # reetiquetamos la población con diabetes como "1"
        df["Diabetes_012"].replace({2: 1}, inplace= True)
        df.drop(["PhysHlth", "MentHlth"], axis=1, inplace=True)
        # y = df_new[['Diabetes_012']]
        X = df.drop(['Diabetes_012'], axis=1)
 
        feature_names = X.columns

        with st.form("Formulario:"):

                col1, col2, col3  = st.columns(3)
                with col1:
                
                        HighBP = st.radio(label = "¿Tienes la presión arterial alta?", 
                                        options = ("Si", "No"), 
                                        index = 1,
                                        disabled = False,
                                        horizontal = True,
                                )
                
                with col2:
                        HighChol = st.radio(label= "¿Tienes el colesterol alto?",
                                        options = ("Si", "No"), 
                                        index = 1,
                                        disabled = False,
                                        horizontal = True,
                                )
                with col3:
                        CholCheck = st.radio(label= "¿Haces análisis de colesterol anuales?",
                                        options = ("Si", "No"), 
                                        index = 1,
                                        disabled = False,
                                        horizontal = True,
                                )
                        
                col4, col5, col6  = st.columns(3)
                with col4:
                        Smoker = st.radio(label= "¿Eres fumador habitual?",
                                        options = ("Si", "No"), 
                                        index = 1,
                                        disabled = False,
                                        horizontal = True,
                                )
                with col5:
                        Stroke = st.radio(label= "¿Has tenido algún ataque al corazón?",
                                        options = ("Si", "No"), 
                                        index = 1,
                                        disabled = False,
                                        horizontal = True,
                                )
                
                with col6:
                        HeartDiseaseorAttack = st.radio(label= "¿Padeces alguna enfermedad cardiovascular?",
                                        options = ("Si", "No"), 
                                        index = 1,
                                        disabled = False,
                                        horizontal = True,
                                )
                        
                col7, col8, col9  = st.columns(3)
                with col7:
                        PhysActivity = st.radio(label= "¿Haces ejercicio con frecuencia?",
                                        options = ("Si", "No"), 
                                        index = 1,
                                        disabled = False,
                                        horizontal = True,
                                )
                with col8:
                        Fruits = st.radio(label= "¿Comes frutas habitualmente?",
                                        options = ("Si", "No"), 
                                        index = 1,
                                        disabled = False,
                                        horizontal = True,
                                )
                
                with col9:
                        Veggies = st.radio(label= "¿Consumes vegetales habitualmente?",
                                        options = ("Si", "No"), 
                                        index = 1,
                                        disabled = False,
                                        horizontal = True,
                                )
                        
                col10, col11, col12  = st.columns(3)
                with col10:
                        HvyAlcoholConsump = st.radio(label= "¿Bebes demasiado alcohol con frecuencia?",
                                        options = ("Si", "No"), 
                                        index = 1,
                                        disabled = False,
                                        horizontal = True,
                                )
                with col11:
                        AnyHealthcare = st.radio(label= "¿Cuidas tu salud?",
                                        options = ("Si", "No"), 
                                        index = 1,
                                        disabled = False,
                                        horizontal = True,
                                )
                with  col12:
                        DiffWalk = st.radio(label= "¿Tienes dificultades para caminar?",
                                        options = ("Si", "No"),  
                                        index = 1,
                                        disabled = False,
                                        horizontal = True,
                                )
                        
                col13, col14, col15  = st.columns(3)
                with col13:
                        NoDocbcCost = st.radio(label= "¿Puedes pagar la asistencia médica?",
                                        options = ("Si", "No"), 
                                        index = 1,
                                        disabled = False,
                                        horizontal = True,
                                )
                with col14:
                        Sex = st.radio(label= "Selecciona tu sexo",
                                        options = ("Mujer", "Hombre"), 
                                        index = 0,
                                        disabled = False,
                                        horizontal = True,
                                )
                with col15:
                        Age = st.number_input(label = "¿Cuántos años tienes?", 
                                #     placeholder = "introduce tu edad",
                                value = 30
                                ) 
                        
                col16, col17, col18 = st.columns(3)
                with col16:
                        GenHlth = st.radio(label= "¿Cómo evaluarías tu estado general de salud del 1 al 5?",
                                        options = ("1", "2", "3", "4", "5"), 
                                        index = 1,
                                        disabled = False,
                                        horizontal = True,
                                )
                with col17:
                        Education = st.radio(label= "¿Cúal es tu nivel educativo del 1 al 6?",
                                        options = ("1", "2", "3", "4", "5", "6"), 
                                        index = 1,
                                        disabled = False,
                                        horizontal = True,
                                )
                with col18:
                        Income = st.radio(label= "¿Cúal es tu franja de ingresos salariales del 1 al 8?",
                                        options = ("1", "2", "3", "4", "5", "6", "7", "8"), 
                                        index = 3,
                                        disabled = False,
                                        horizontal = True,
                                )
                        
                col19, col20, col21 = st.columns(3)
                with col19:
                        peso = st.number_input(label = "Calculadora IMC: Introduce tu peso en kg", 
                                value = 60
                                ) 
                with col20:
                        altura = st.number_input(label = "Calculadora IMC: Introduce tu altura en cm",
                                value = 160
                                ) 
                IMC = round(peso/(altura/100)**2)
                with col21:
                        BMI = st.number_input(label = "Resultado IMC tu índice de masa corporal es:", 
                                        # placeholder = "introduce tu IMC",
                                        value = IMC
                                )

                if st.form_submit_button("Enviar"):
                        datos = [HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, GenHlth, DiffWalk, Sex, Age, Education, Income ]

                        for i in range(len(datos)):

                                if datos[i] == 'Si':
                                        datos[i] = 1
                                        
                                elif datos[i] == 'No':
                                        datos[i] = 0
                                
                                elif datos[i] == 'Mujer':
                                        datos[i] = 0
                                        
                                elif datos[i] == 'Hombre':
                                        datos[i] = 1

                        dfdatos = pd.DataFrame(data = [datos], columns = feature_names)

                        datos1 = {"Column2": "example_value"}
                        datos2 = dfdatos.iloc[0].to_dict()

                        for clave,valor in datos2.items():
                                datos1[clave]=valor

                        datos =  {
                                "Inputs": {
                                        "data": [
                                                datos1
                                                ]
                                        },
                                "GlobalParameters": {
                                                "method": "predict"
                                                }
                                }
                        
                        prediction = azresult(datos)
                                        
                        resultado = json.loads(prediction)

                        valor = resultado["Results"][0]
                
                        if valor == 1:
                                st.error("Según nuestro modelo es probable que tengas **diabetes**. Consulta a tu médico.")
                        else:
                                st.success("Según nuestro modelo **no estás en riesgo** de tener diabetes")
                        
                        col22, col23 = st.columns(2)
                        with col22:
                                st.write("% de probabilidad de no padecer diabetes:")
                                
                                datos_proba = datos.copy()

                                datos_proba["GlobalParameters"]["method"] = "predict_proba"
                                prediction_proba = azresult(datos_proba)
                                        
                                resultado_proba = json.loads(prediction_proba)

                                st.write(int(resultado_proba["Results"][0][0]*100))

                        with col23:
                                st.write("% de probabilidad de ser diabético:")
                                st.write(int(resultado_proba["Results"][0][1]*100))

    with tab3:
        
        st.title("Drug Analyzer")

        #st.image("data/drugs.png")

        train = pd.read_csv('data/drugsComTrain_raw.tsv',sep ="\t")
        test = pd.read_csv('data/drugsComTest_raw.tsv',sep ="\t")
        df = pd.concat([train, test])
        df = df.drop(["Unnamed: 0", "date"], axis=1)
        df_sintomas = pd.read_excel('data/tipo_enfermedad.xlsx')
        df_drugs = pd.read_excel('data/principio_activo.xlsx')

        # EDA

        df.rating = df.rating.astype(int)
        df.reset_index(inplace = True, drop = True)
        df.drop_duplicates(subset=["review"], inplace = True)
        df.condition.fillna("EMPTY", inplace = True)
        #df.drop(columns = ["date"], inplace = True)
        df["review"] = df["review"].str.replace('&#039;', '').str.replace("&amp;", '').str.replace('\r\n', '').str.replace('&quot;', '').str.replace("'" , '').str.replace("`" , '').str.replace('"' , '').str.replace('-' , '').str.replace("cance", "cancer")
        #st.dataframe(df)

        ################################################

        #filter the principio activo poniendo enfermedades
        search_word =st.selectbox("Tipo de enfermedad: ",options=[""]+", ".join(df_sintomas.columns).split(", "))

        filtered_columns = df_sintomas.columns[df_sintomas.columns.str.contains(search_word, case=True)]
        filtered_df = df_sintomas[filtered_columns]

        #st.write(filtered_df)
        st.dataframe(filtered_df)
        #st.table(filtered_df)

        #########################################################
        ### filter principio activo en lista de medicamentos
        
        activo = st.text_input('Nombre del principio Activo: ')

        filtered_dfactivo = df_drugs[df_drugs.apply(lambda row: row.astype(str).str.contains(activo, case=False).any(), axis=1)]

        st.dataframe(filtered_dfactivo, width = 880)

        ###########################################################################3
        detalles = st.text_input('Indica el Nº Registro del medicamento que interesa: ')

        filtered_dfactivos3 = df_drugs[df_drugs["Nº Registro"].str.contains(detalles, case=False)]
        st.write(filtered_dfactivos3)
        
        foto_url = 'https://cima.aemps.es/cima/fotos/full/materialas/{}/{}_materialas.jpg'.format(detalles, detalles)

        response = requests.get(foto_url)

        if response.status_code == 200:
                response.raise_for_status()
                image = foto_url
                st.image(image=foto_url, use_column_width=True)

        else:
                st.write('No hay foto disponible.')

        url = f"https://cima.aemps.es:443/cima/dochtml/ft/{detalles}/FT_{detalles}.html"

        # Create a link to the URL
        st.markdown(f"Click [Aqui]({url}) para ver mas información acerca de este medicamento.",
                unsafe_allow_html=True)

        redirect_js = f"window.location.href = '{url}'; return false;"


        ##################################


        st.title("Opiniones de los consumidores")
        st.write("¿Quieres saber qué opinan acerca del medicamento que quieres comprar?")

        #filter1_col = df["drugName"].unique()
        filter1_val = st.text_input("Introducir el nombre del medicamento:", '')

        filtered_df = df[df['drugName'] == filter1_val]

        st.dataframe(filtered_df, width = 880)

        #################################33

        with st.form("Deja tu opinión"):
                texto_rating = st.text_input("Valora este medicamento de 1 a 10: ")

                texto = st.text_area(label = "Por favor deje su comentario acerca de este medicamento", 
                                height = 150, 
                                max_chars = 2000,
                                placeholder = "Comentario")

                # Every form must have a submit button.
                submitted = st.form_submit_button("Submit")
        if submitted:
                st.write(f"Su mensaje ha sido enviado correctamente")
                df = df.append({'drugName': filter1_val, 'review': texto, 'rating': texto_rating}, ignore_index=True)

        # Print the updated DataFrame
        st.dataframe(df[['drugName', 'review', 'rating']].tail(5), width = 880)

    with tab4:
        st.image("data/Spanish_plate.webp")
        st.markdown("Todos los posibles análisis siempre conducen a la misma conclusión. **:blue[Implementar estilos de vida saludables, alimentarse adecuadamente y de forma variada ayuda a prevenir enfermedades.]**")        
        st.markdown("“Derechos de autor © 2011 Universidad de Harvard. Para más información sobre El Plato para Comer Saludable, por favor visite la Fuente de Nutrición, Departamento de Nutrición, Escuela de Salud Pública de Harvard, http://www.thenutritionsource.org y Publicaciones de Salud de Harvard, health.harvard.edu.”")

if __name__ == "__main__":
                main()