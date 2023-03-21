import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from IPython.display import display, Image
import pickle

def main():
    st.image("data/portada.png", width=880)

    tab0, tab1, tab2, tab3, tab4 = st.tabs(["Intro Health Watcher", "Estudio de Enfermedades con power BI", "Predicción de  Diabetes con Machine Learning", "Drug Analyzer", "Recomendaciones"])

    with tab0:
        st.image("data/drugs.png")
        st.write("Análisis de enfermedades y medicamentos con streamlit, power BI, sklearn.")

        st.markdown("""Los datos de este proyectos proceden de diversas fuentes públicas : 
                          [**CIMA**](https://cima.aemps.es/cima/publico/home.html), [**Drugs**](https://drugs.com), [**INE**](https://datos.gob.es/es/catalogo?publisher_display_name=Instituto+Nacional+de+Estad%C3%ADstica)""")

        st.write("""**`Estudio de Enfermedades con power BI`**  muestra estadísticas sobre enfermedades.""")

        st.write("""**`Predicción de  Diabetes con Machine Learning`**  calcula su probabilidad de contraer diabetes contestando una pequeña encuesta sobre sus hábitos de vida saludables con una sensibilidad del 90%.""")

        st.write("""**`Drug Analyzer`**  proporciona información sobre medicamentos.""")

        st.error("""Nota: toda la información proporcionada por esta app es orientativa y basada en datos públicos. Recuerde que ante cualquier duda siempre debe acudir a su médico de cabecera.""")

    with tab1:

        st.components.v1.html('''<iframe title="gastos_comunidad" width="880" height="550" src="https://app.powerbi.com/view?r=eyJrIjoiNDVmMGNiZGItZjZhMS00MDQ5LWIwMGEtNjE2ZWIyYjYxMThjIiwidCI6IjYzOTc4MWU1LWJjMmYtNGE3Ni04YmY1LTJiNTg0ZDcxN2U5ZCIsImMiOjl9&pageName=ReportSection" frameborder="0" allowFullScreen="true"></iframe>''', width = 880, height = 580)
 
    with tab2:
        st.header("ML Predicción de Diabetes según Encuesta")
        st.caption("Recuerda que esta app es sólo informativa, ante cualquier duda médica siempre debes acudir a tu médico de cabecera o consultar a un profesional.")
        st.caption("Modelo de ML RandomForest con un Recall del 89%")
                 
        df = pd.read_csv("../data/diabetes_012_health_indicators_BRFSS2015.csv")
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
        
        # calculadora de IMC
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
               
        
        
        data = [HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, GenHlth, DiffWalk, Sex, Age, Education, Income ]
       
        for i in range(len(data)):

                if data[i] == 'Si':
                        data[i] = 1
                        
                elif data[i] == 'No':
                        data[i] = 0
                
                elif data[i] == 'Mujer':
                        data[i] = 0
                        
                elif data[i] == 'Hombre':
                        data[i] = 1

        df_usuario = pd.DataFrame(data = [data], columns = feature_names)
        # st.dataframe(df_usuario)
 
        with open("data/escalador.sav", "rb") as file:
                scaler = pickle.load(file)

        df_usu = scaler.transform(df_usuario)

        # st.dataframe(df_usu)

        with open("data/modela_rf.sav", "rb") as file:
                model = pickle.load(file)

        prediction = model.predict(df_usuario)
        if prediction == 1:
               st.error("Según nuestro modelo es probable que tengas **diabetes**. Consulta a tu médico.")
        else:
               st.success("Según nuestro modelo **no estás en riesgo** de tener diabetes")
        
        col22, col23 = st.columns(2)
        with col22:
                st.write("% probabilidad de no padecer diabetes:")
                st.write(int(model.predict_proba(df_usuario)[0][0]*100))
        with col23:
                st.write("% probabilidad de ser diabético:")
                st.write(int(model.predict_proba(df_usuario)[0][1]*100))

    with tab3:
        st.header("Drug Analyzer")

        train = pd.read_csv('data/drugsComTrain_raw.tsv',sep ="\t")
        test = pd.read_csv('data/drugsComTest_raw.tsv',sep ="\t")
        df = pd.concat([train, test])
        # EDA
        df.rating = df.rating.astype(int)
        df.reset_index(inplace = True, drop = True)
        df.drop_duplicates(subset=["review"], inplace = True)
        df.condition.fillna("EMPTY", inplace = True)
        df.drop(columns = ["Unnamed: 0", "date"], inplace = True)
        df["condition"] = df["condition"].str.replace("cance","cancer")
        df["review"] = df["review"].str.replace('&#039;', '').str.replace("&amp;", '').str.replace('\r\n', '').str.replace('&quot;', '').str.replace("'" , '').str.replace("`" , '').str.replace('"' , '').str.replace('-' , '')
        with st.expander(label = "DataFrame Drugs", expanded = False):
                st.dataframe(df)

        # Get user input
        search_word = st.text_input('Tipo de enfermedad:', '')

        # Filter the DataFrame based on the search word
        filtered_df = df[df['condition'].str.contains(search_word)]

        st.write('Results:', len(filtered_df))
        st.write(filtered_df)

        # Get user input for first filter
        col21, col22 = st.columns(2)
        with col21:
               filter1_col = st.selectbox('Select a column to filter by:', df.columns)
        with col22:
               filter1_val = st.text_input('Enter a value to filter by:', '')

        # Filter the DataFrame based on the first filter
        
        if filter1_val:
                filtered_df = df[df[filter1_col] == filter1_val]
        else:
                filtered_df = df

        # Get user input for second filter
        col23, col24 = st.columns(2)
        with col23:
               filter2_col = st.selectbox('Select a second column to filter by:', filtered_df.columns)
        with col24:
               filter2_val = st.text_input('Enter a second value to filter by:', '')

        # Filter the DataFrame based on the second filter
        if filter2_val:
                filtered_df = filtered_df[filtered_df[filter2_col] == filter2_val]

        # Display the filtered DataFrame
        st.write(filtered_df)

        medicamento = st.text_input(label = "Name", 
                        max_chars = 20,
                        placeholder = "Nombre del Medicamento")
        
        st.title(medicamento)

        url = f"https://www.drugs.com/{medicamento}.html"
        

        st.write("Puede ver más información en:", url)

        # Desde aquí df de medicamentos
        df_drugs = pd.read_excel('data/Medicamentos.xls')

        filtered_dfactivos2 = df_drugs[df_drugs["Medicamento"].str.contains(medicamento, case=False)]
        filtered_dfactivos3 = df_drugs[df_drugs["Principios Activos"].str.contains(medicamento, case=False)]

        dfs = pd.concat([filtered_dfactivos2 , filtered_dfactivos3 ])

        detalles = dfs["Nº Registro"]
        detalles = detalles.reset_index(drop = True)

        # url2 =https://cima.aemps.es/cima/publico/detalle.html?nregistro={detalles[0]}

        # detalles = input('Indica el No Registro del medicamento que interesa: ')
        lista_fotos = list()

        for i in range(len(detalles)):
            
            foto_url = 'https://cima.aemps.es/cima/fotos/full/materialas/{}/{}_materialas.jpg'.format(detalles[i], detalles[i])
            lista_fotos.append(foto_url)
            (display(Image(url=foto_url)))
        st.image(lista_fotos[0], width=400)   
        st.dataframe(filtered_dfactivos2)


        # Define the URL to redirect to buscar el prospecto en otro df
        redirect_url = ('https://cima.aemps.es/cima/fotos/full/materialas/{}/{}_materialas.jpg').format(detalles[0], detalles[0])

        # Create a link to the URL
        st.markdown(f"Click [Aqui]({redirect_url}) para ver mas información acerca de este medicamento.")

        number = st.number_input(label = "Valorar este medicamento de 1 a 10",
                                min_value = 0,
                                max_value = 10,
                                value = 0,
                                step = 1)

        texto = st.text_area(label = "Por favor dejar su comentario acerca de este medicamento", 
                                height = 150, 
                                max_chars = 2000,
                                placeholder = "Comentario")
        st.write(texto)

        if st.button(label = "Submit",
                        key   = "submit2",
                        type  = "primary"):
                st.write(f"Su mensaje ha sido enviado corretamente")


    with tab4:
        st.image("data/Spanish_plate.webp")
        st.markdown("Todos los posibles análisis siempre conducen a la misma conclusión. **:blue[Implementar estilos de vida saludables, alimentarse adecuadamente y de forma variada ayuda a prevenir enfermedades.]**")        
        st.markdown("“Derechos de autor © 2011 Universidad de Harvard. Para más información sobre El Plato para Comer Saludable, por favor visite la Fuente de Nutrición, Departamento de Nutrición, Escuela de Salud Pública de Harvard, http://www.thenutritionsource.org y Publicaciones de Salud de Harvard, health.harvard.edu.”")

if __name__ == "__main__":
                main()