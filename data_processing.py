# =========================================
# DATA PROCESSING - FILE ORGANIZER
# =========================================

from langchain_community.document_loaders import PyPDFLoader
import os


# state
def classify_by_state(text: str) -> str:
    estados = ["New York", "Florida", "California"]
    for e in estados:
        if e.lower() in text.lower():
            return e
    return "Unknown"


# read all pdf docuemnts and detect whish US state
# reciebe 2 rutas: input = recibe sin clasifica, output: se crea subcarpetas po estado
def classify_documents(input_folder: str, output_folder: str):
    for file in os.listdir(input_folder):  # recorre archivos dentro de carpeta
        if file.endswith(".pdf"):
            file_path = os.path.join(input_folder, file)
            # PyPDFloader
            loader = PyPDFLoader(file_path)
            docs = loader.load()  # carga documento
            # extraer texto principal del primer documento
            if len(docs) > 0:
                text = " ".join([doc.page_content for doc in docs])

            # LLAMAR FUNCION EXTERNA
            estado = classify_by_state(text)
            dest_folder = os.path.join(
                output_folder, estado
            )  # crear la ruta de destino
            os.makedirs(dest_folder, exist_ok=True)  # si ya existe no la crea
            shutil.move(os.path.join(input_folder, file), dest_folder)


input_folder = r"C:\Users\grupo\OneDrive\Escritorio\HACKATHON\data_insurance"
output_folder = (
    r"C:\Users\grupo\OneDrive\Escritorio\HACKATHON\data_insurance\classified_data"
)
sample_docs = classify_documents(input_folder, output_folder)
