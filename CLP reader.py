import fitz  # PyMuPDF
import cv2
import numpy as np
import PySimpleGUI as sg
import io
from math import sqrt
import re
import google.generativeai as genai
import os
import markdown2

genai.configure(api_key = 'AIzaSyB3G1_u8PuzxTr5igvAOlW0xncgWVKMTZk')
model = genai.GenerativeModel('gemini-pro')

model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])


def extract_text_and_images_from_pdf(pdf_path, dpi=300):
    doc = fitz.open(pdf_path)
    text = ""
    images = []
    for page in doc:
        page_text = page.get_text()
        if page_text.strip():  # Assurez-vous qu'il y a du contenu dans la page
            text += page_text.strip() + '\n'  # Ajoutez le texte de la page avec un retour à la ligne
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        images.append(img_bgr)
    return text, images, dpi

def extract_and_calculate_packaging_area(text):
    # Pattern to match the "125 x 110 mm" format, allowing for optional spaces around 'x' and case-insensitive 'mm'
    pattern = r'\b(\d+)\s*x\s*(\d+)\s*mm\b'
    match = re.search(pattern, text, re.IGNORECASE)  # Use the re.IGNORECASE flag to ignore case
    if match:
        # Extract the width and height as integers
        width = int(match.group(1))
        height = int(match.group(2))
        # Calculate the area
        area = width * height
        # Return only the area in square millimeters
        return area
    return None  # Return None if packaging size not found

def find_smallest_bounding_box_and_measure(images, dpi):
    largest_dimensions = (0, 0)  # Initialize with 0 width and 0 height
    largest_area = 0
    image_with_largest_box = None
    largest_box = None  # To store the coordinates of the largest box

    for image in images:
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply threshold to create a binary image where non-white areas are black
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

        # Find contours in the binary image
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the bounding rectangle for the largest contour by area
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            current_area = w * h

            # Check if the current bounding box is the largest found so far
            if current_area > largest_area:
                largest_dimensions = (w, h)
                largest_area = current_area
                image_with_largest_box = image.copy()  # Make a copy of the image to draw on
                largest_box = (x, y, w, h)  # Store the coordinates and dimensions of the largest box

    # Draw the largest bounding box on the corresponding image
    if image_with_largest_box is not None and largest_box is not None:
        x, y, w, h = largest_box
        cv2.rectangle(image_with_largest_box, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Convert the largest bounding box dimensions from pixels to millimeters
        width_mm = (largest_dimensions[0] * 25.4) / dpi
        height_mm = (largest_dimensions[1] * 25.4) / dpi

        return (str(int(width_mm)) + ' x ' + str(int(height_mm)) + 'mm (estimated with detection)')
    else:
        return None, None


def detect_clp_pictogram_and_measure(image, dpi):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Seuils HSV ajustés pour cibler des teintes de rouge spécifiques et éviter RGB (231, 56, 67)
    # Notez que les valeurs exactes pour lower_red1 et upper_red1 doivent être déterminées
    # en convertissant RGB (231, 56, 67) en HSV et en ajustant les seuils pour l'exclure.
    lower_red1 = np.array([0, 100, 100])  # Exemple de seuil bas, ajustez selon les besoins
    upper_red1 = np.array([5, 255, 255])  # Exemple de seuil haut, ajustez selon les besoins

    lower_red2 = np.array([175, 100, 100])  # Autre exemple de seuil bas, ajustez selon les besoins
    upper_red2 = np.array([180, 255, 255])  # Autre exemple de seuil haut, ajustez selon les besoins

    # Créer des masques basés sur les seuils définis
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            cv2.drawContours(image, [approx], -1, (0, 255, 0), 3)
            left = min(approx[:,0,0])
            right = max(approx[:,0,0])
            pixel_distance = (right - left) / sqrt(2)
            size_mm = (pixel_distance * 25.4) / dpi
            return size_mm, image
    return None, None


def extract_volume_from_text(text):
    pattern = r'\b(\d+(\.\d+)?)\s*([mM]?[lL])\b'
    match = re.search(pattern, text, re.IGNORECASE)  # Utiliser le flag re.IGNORECASE pour ignorer la casse
    if match:
        # Retourner uniquement le premier volume trouvé, en normalisant les unités en majuscules pour la cohérence
        return f"{match.group(1)} {match.group(3).upper()}"
    return "Volume not found"

def extract_packaging_size_from_text(text):
    # Pattern to match the "125 x 110 mm" format, allowing for optional spaces around 'x' and case-insensitive 'mm'
    pattern = r'\b(\d+)\s*x\s*(\d+)\s*mm\b'
    match = re.search(pattern, text, re.IGNORECASE)  # Use the re.IGNORECASE flag to ignore case
    if match:
        # Return the matched packaging size, formatted as "Width x Height mm"
        return f"{match.group(1)} x {match.group(2)} mm"
    return "Packaging size not found"


def extract_pack_size_from_text(text):
    pattern = r'\b(\d+(\.\d+)?)\s*([mm]?[mm])\b'
    match = re.search(pattern, text, re.IGNORECASE)  # Utiliser le flag re.IGNORECASE pour ignorer la casse
    if match:
        # Retourner uniquement le premier volume trouvé, en normalisant les unités en majuscules pour la cohérence
        return f"{match.group(1)} {match.group(3).upper()}"
    return "Volume not found"



def convert_image_to_bytes(image):
    is_success, buffer = cv2.imencode(".png", image)
    return io.BytesIO(buffer).getvalue() if is_success else None

def resize_and_convert_image_to_bytes(image, max_width=1600, max_height=900):
    # Calculer le ratio pour maintenir le ratio d'aspect
    height, width = image.shape[:2]
    scale_width = max_width / width
    scale_height = max_height / height
    scale = min(scale_width, scale_height)

    # Appliquer le redimensionnement
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Convertir en bytes
    is_success, buffer = cv2.imencode(".png", resized_image)
    if not is_success:
        return None
    return io.BytesIO(buffer).getvalue()

def classify_text(text):
    # Exemple simple de classification basée sur des mots-clés
    mode_d_emploi = ""
    marketing = ""
    clp = ""
    autres = text  # Par défaut, tout le texte va dans "autres"

    # Imaginons que nous avons des marqueurs simples pour identifier chaque section
    if "Mode d'emploi:" in text:
        mode_d_emploi = "Votre texte mode d'emploi ici..."
        autres = autres.replace(mode_d_emploi, '')  # Exclure du texte 'autres'
    if "Marketing:" in text:
        marketing = "Votre texte marketing ici..."
        autres = autres.replace(marketing, '')

    # Supprimer les espaces entre les mots dans le texte
    text = text.replace(' ', '')

    phrases = set()  # Pour stocker les phrases CLP uniques

    if "Explosifinstable." in text:
        phrases.add("H200: Explosif instable.")
        autres = autres.replace("Explosif instable.", '')

    if "Explosif;dangerd'explosionenmasse." in text:
        phrases.add("H201: Explosif; danger d'explosion en masse.")
        autres = autres.replace("Explosif; danger d'explosion en masse.", '')

    if "Explosif;dangersérieuxdeprojection." in text:
        phrases.add("H202: Explosif; danger sérieux de projection.")
        autres = autres.replace("Explosif; danger sérieux de projection.", '')

    if "Explosif;dangerd'incendie,d'effetdesouffleoudeprojection." in text:
        phrases.add("H203: Explosif; danger d'incendie, d'effet de souffle ou de projection.")
        autres = autres.replace("Explosif; danger d'incendie, d'effet de souffle ou de projection.", '')

    if "Dangerd'incendieoudeprojection." in text:
        phrases.add("H204: Danger d'incendie ou de projection.")
        autres = autres.replace("Danger d'incendie ou de projection.", '')

    if "Dangerd'explosionenmasseencasd'incendie." in text:
        phrases.add("H205: Danger d'explosion en masse en cas d'incendie.")
        autres = autres.replace("Danger d'explosion en masse en cas d'incendie.", '')

    if "Gazextrêmementinflammable." in text:
        phrases.add("H220: Gaz extrêmement inflammable.")
        autres = autres.replace("Gaz extrêmement inflammable.", '')

    if "Gazinflammable." in text:
        phrases.add("H221: Gaz inflammable.")
        autres = autres.replace("Gaz inflammable.", '')

    if "Aérosolextrêmementinflammable." in text:
        phrases.add("H222: Aérosol extrêmement inflammable.")
        autres = autres.replace("Aérosol extrêmement inflammable.", '')

    if "Aérosolinflammable." in text:
        phrases.add("H223: Aérosol inflammable.")
        autres = autres.replace("Aérosol inflammable.", '')

    if "Liquideetvapeursextrêmementinflammables." in text:
        phrases.add("H224: Liquide et vapeurs extrêmement inflammables.")
        autres = autres.replace("Liquide et vapeurs extrêmement inflammables.", '')

    if "Liquideetvapeurstrèsinflammables." in text:
        phrases.add("H225: Liquide et vapeurs très inflammables.")
        autres = autres.replace("Liquide et vapeurs très inflammables.", '')

    if "Liquideetvapeursinflammables." in text:
        phrases.add("H226: Liquide et vapeurs inflammables.")
        autres = autres.replace("Liquide et vapeurs inflammables.", '')

    if "Matièresolideinflammable." in text:
        phrases.add("H228: Matière solide inflammable.")
        autres = autres.replace("Matière solide inflammable.", '')

    if "Peutexplosersousl'effetdelachaleur." in text:
        phrases.add("H240: Peut exploser sous l'effet de la chaleur.")
        autres = autres.replace("Peut exploser sous l'effet de la chaleur.", '')

    if "Peuts'enflammerouexplosersousl'effetdelachaleur." in text:
        phrases.add("H241: Peut s'enflammer ou exploser sous l'effet de la chaleur.")
        autres = autres.replace("Peut s'enflammer ou exploser sous l'effet de la chaleur.", '')

    if "Peuts'enflammersousl'effetdelachaleur." in text:
        phrases.add("H242: Peut s'enflammer sous l'effet de la chaleur.")
        autres = autres.replace("Peut s'enflammer sous l'effet de la chaleur.", '')

    if "S'enflammespontanémentaucontactdel'air." in text:
        phrases.add("H250: S'enflamme spontanément au contact de l'air.")
        autres = autres.replace("S'enflamme spontanément au contact de l'air.", '')

    if "Matièreauto-échauffante;peuts'enflammer." in text:
        phrases.add("H251: Matière auto-échauffante; peut s'enflammer.")
        autres = autres.replace("Matière auto-échauffante; peut s'enflammer.", '')

    if "Matièreauto-échauffanteengrandesquantités;peuts'enflammer." in text:
        phrases.add("H252: Matière auto-échauffante en grandes quantités; peut s'enflammer.")
        autres = autres.replace("Matière auto-échauffante en grandes quantités; peut s'enflammer.", '')

    if "Dégageaucontactdel'eaudesgazinflammablesquipeuvents'enflammerspontanément." in text:
        phrases.add("H260: Dégage au contact de l'eau des gaz inflammables qui peuvent s'enflammer spontanément.")
        autres = autres.replace("Dégage au contact de l'eau des gaz inflammables qui peuvent s'enflammer spontanément.", '')

    if "Dégageaucontactdel'eaudesgazinflammables." in text:
        phrases.add("H261: Dégage au contact de l'eau des gaz inflammables.")
        autres = autres.replace("Dégage au contact de l'eau des gaz inflammables.", '')

    if "Peutprovoquerouaggraverunincendie;comburant." in text:
        phrases.add("H270: Peut provoquer ou aggraver un incendie; comburant.")
        autres = autres.replace("Peut provoquer ou aggraver un incendie; comburant.", '')

    if "Peutprovoquerunincendieouuneexplosion;comburantpuissant." in text:
        phrases.add("H271: Peut provoquer un incendie ou une explosion; comburant puissant.")
        autres = autres.replace("Peut provoquer un incendie ou une explosion; comburant puissant.", '')

    if "Peutaggraverunincendie;comburant." in text:
        phrases.add("H272: Peut aggraver un incendie; comburant.")
        autres = autres.replace("Peut aggraver un incendie; comburant.", '')

    if "Contientungazsouspression;peutexplosersousl'effetdelachaleur." in text:
        phrases.add("H280: Contient un gaz sous pression; peut exploser sous l'effet de la chaleur.")
        autres = autres.replace("Contient un gaz sous pression; peut exploser sous l'effet de la chaleur.", '')

    if "Contientungazréfrigéré;peutcauserdesbrûluresoublessurescryogéniques." in text:
        phrases.add("H281: Contient un gaz réfrigéré; peut causer des brûlures ou blessures cryogéniques.")
        autres = autres.replace("Contient un gaz réfrigéré; peut causer des brûlures ou blessures cryogéniques.", '')

    if "Peutêtrecorrosifpourlesmétaux." in text:
        phrases.add("H290: Peut être corrosif pour les métaux.")
        autres = autres.replace("Peut être corrosif pour les métaux.", '')

    if "Mortelencasd’ingestion." in text:
        phrases.add("H300: Mortel en cas d’ingestion.")
        autres = autres.replace("Mortel en cas d’ingestion.", '')

    if "Toxiqueencasd’ingestion." in text:
        phrases.add("H301: Toxique en cas d’ingestion.")
        autres = autres.replace("Toxique en cas d’ingestion.", '')

    if "Nocifencasd’ingestion." in text:
        phrases.add("H302: Nocif en cas d’ingestion.")
        autres = autres.replace("Nocif en cas d’ingestion.", '')

    if "Peutêtremortelencasd’ingestionetdepénétrationdanslesvoiesrespiratoires." in text:
        phrases.add("H304: Peut être mortel en cas d’ingestion et de pénétration dans les voies respiratoires.")
        autres = autres.replace("Peut être mortel en cas d’ingestion et de pénétration dans les voies respiratoires.", '')

    if "Mortelparcontactcutané." in text:
        phrases.add("H310: Mortel par contact cutané.")
        autres = autres.replace("Mortel par contact cutané.", '')

    if "Toxiqueparcontactcutané." in text:
        phrases.add("H311: Toxique par contact cutané.")
        autres = autres.replace("Toxique par contact cutané.", '')

    if "Nocifparcontactcutané." in text:
        phrases.add("H312: Nocif par contact cutané.")
        autres = autres.replace("Nocif par contact cutané.", '')

    if "Provoquedesbrûluresdelapeauetdegraveslésionsdesyeux." in text:
        phrases.add("H314: Provoque des brûlures de la peau et de graves lésions des yeux.")
        autres = autres.replace("Provoque des brûlures de la peau et de graves lésions des yeux.", '')

    if "Provoqueuneirritationcutanée." in text:
        phrases.add("H315: Provoque une irritation cutanée.")
        autres = autres.replace("Provoque une irritation cutanée.", '')

    if "Peutprovoqueruneallergiecutanée." in text:
        phrases.add("H317: Peut provoquer une allergie cutanée.")
        autres = autres.replace("Peut provoquer une allergie cutanée.", '')

    if "Provoquedegraveslésionsdesyeux." in text:
        phrases.add("H318: Provoque de graves lésions des yeux.")
        autres = autres.replace("Provoque de graves lésions des yeux.", '')

    if "Provoqueunegraveirritationoculaire." in text:
        phrases.add("H319: Provoque une grave irritation oculaire.")
        autres = autres.replace("Provoque une grave irritation oculaire.", '')

    if "Mortelparinhalation." in text:
        phrases.add("H330: Mortel par inhalation.")
        autres = autres.replace("Mortel par inhalation.", '')

    if "Toxiqueparinhalation." in text:
        phrases.add("H331: Toxique par inhalation.")
        autres = autres.replace("Toxique par inhalation.", '')

    if "Nocifparinhalation." in text:
        phrases.add("H332: Nocif par inhalation.")
        autres = autres.replace("Nocif par inhalation.", '')

    if "Peutprovoquerdessymptômesallergiquesoud’asthmeoudesdifficultésrespiratoiresparinhalation." in text:
        phrases.add("H334: Peut provoquer des symptômes allergiques ou d’asthme ou des difficultés respiratoires par inhalation.")
        autres = autres.replace("Peut provoquer des symptômes allergiques ou d’asthme ou des difficultés respiratoires par inhalation.", '')

    if "Peutirriterlesvoiesrespiratoires." in text:
        phrases.add("H335: Peut irriter les voies respiratoires.")
        autres = autres.replace("Peut irriter les voies respiratoires.", '')

    if "Peutprovoquersomnolenceouvertiges." in text:
        phrases.add("H336: Peut provoquer somnolence ou vertiges.")
        autres = autres.replace("Peut provoquer somnolence ou vertiges.", '')

    if "Peutinduiredesanomaliesgénétiques<indiquerlavoied’expositions’ilestformellementprouvéqu’aucuneautrevoied’expositionneconduitaumêmedanger>." in text:
        phrases.add("H340: Peut induire des anomalies génétiques <indiquer la voie d’exposition s’il est formellement prouvé qu’aucune autre voie d’exposition ne conduit au même danger>.")
        autres = autres.replace("Peut induire des anomalies génétiques <indiquer la voie d’exposition s’il est formellement prouvé qu’aucune autre voie d’exposition ne conduit au même danger>.", '')

    if "Susceptibled’induiredesanomaliesgénétiques<indiquerlavoied’expositions’ilestformellementprouvéqu’aucuneautrevoied’expositionneconduitaumêmedanger>." in text:
        phrases.add("H341: Susceptible d’induire des anomalies génétiques <indiquer la voied’exposition s’il est formellement prouvé qu’aucune autre voie d’exposition ne conduit au même danger>.")
        autres = autres.replace("Susceptible d’induire des anomalies génétiques <indiquer la voied’exposition s’il est formellement prouvé qu’aucune autre voie d’exposition ne conduit au même danger>.", '')

    if "Peutprovoquerlecancer<indiquerlavoied’expositions’ilestformellementprouvéqu’aucuneautrevoied’expositionneconduitaumêmedanger>." in text:
        phrases.add("H350: Peut provoquer le cancer <indiquer la voie d’exposition s’il est formellement prouvé qu’aucune autre voie d’exposition ne conduit au même danger>.")
        autres = autres.replace("Peut provoquer le cancer <indiquer la voie d’exposition s’il est formellement prouvé qu’aucune autre voie d’exposition ne conduit au même danger>.", '')

    if "Susceptibledeprovoquerlecancer<indiquerlavoied'expositions'ilestformellementprouvéqu'aucuneautrevoied'expositionneconduitaumêmedanger>." in text:
        phrases.add("H351: Susceptible de provoquer le cancer <indiquer la voie d'exposition s'il est formellement prouvé qu'aucune autre voie d'exposition ne conduit au même danger>.")
        autres = autres.replace("Susceptible de provoquer le cancer <indiquer la voie d'exposition s'il est formellement prouvé qu'aucune autre voie d'exposition ne conduit au même danger>.", '')

    if "Peutnuireàlafertilitéouaufoetus<indiquerl'effetspécifiques'ilestconnu><indiquerlavoied'expositions'ilestformellementprouvéqu'aucuneautrevoied'expositionneconduitaumêmedanger>." in text:
        phrases.add("H360: Peut nuire à la fertilité ou au foetus <indiquer l'effet spécifique s'il est connu> <indiquer la voie d'exposition s'il est formellement prouvé qu'aucune autre voie d'exposition ne conduit au même danger>.")
        autres = autres.replace("Peut nuire à la fertilité ou au foetus <indiquer l'effet spécifique s'il est connu> <indiquer la voie d'exposition s'il est formellement prouvé qu'aucune autre voie d'exposition ne conduit au même danger>.", '')

    if "Susceptibledenuireàlafertilitéouaufoetus<indiquerl'effets'ilestconnu><indiquerlavoied'expositions'ilestformellementprouvéqu'aucuneautrevoied'expositionneconduitaumêmedanger>." in text:
        phrases.add("H361: Susceptible de nuire à la fertilité ou au foetus <indiquer l'effet s'il est connu> <indiquer la voie d'exposition s'il est formellement prouvé qu'aucune autre voie d'exposition ne conduit au même danger>.")
        autres = autres.replace("Susceptible de nuire à la fertilité ou au foetus <indiquer l'effet s'il est connu> <indiquer la voie d'exposition s'il est formellement prouvé qu'aucune autre voie d'exposition ne conduit au même danger>.", '')

    if "Peutêtrenocifpourlesbébésnourrisaulaitmaternel." in text:
        phrases.add("H362: Peut être nocif pour les bébés nourris au lait maternel.")
        autres = autres.replace("Peut être nocif pour les bébés nourris au lait maternel.", '')

    if "Risqueavéréd'effetsgravespourlesorganes<ouindiquertouslesorganesaffectés,s'ilssontconnus><indiquerlavoied'expositions'ilestformellementprouvéqu'aucuneautrevoied'expositionneconduitaumêmedanger>." in text:
        phrases.add("H370: Risque avéré d'effets graves pour les organes <ou indiquer tous les organes affectés, s'ils sont connus> <indiquer la voie d'exposition s'il est formellement prouvé qu'aucune autre voie d'exposition ne conduit au même danger>.")
        autres = autres.replace("Risque avéré d'effets graves pour les organes <ou indiquer tous les organes affectés, s'ils sont connus> <indiquer la voie d'exposition s'il est formellement prouvé qu'aucune autre voie d'exposition ne conduit au même danger>.", '')

    if "Risqueprésuméd'effetsgravespourlesorganes" in text:
        phrases.add("H371: Risque présumé d'effets graves pour les organes <ou indiquer tous les organes affectés, s'ils sont connus> <indiquer la voie d'exposition s'il est formellement prouvé qu'aucune autre voie d'exposition ne conduit au même danger>.")
        autres = autres.replace("Risque présumé d'effets graves pour les organes <ou indiquer tous les organes affectés, s'ils sont connus> <indiquer la voie d'exposition s'il est formellement prouvé qu'aucune autre voie d'exposition ne conduit au même danger>.", '')

    if "Risqueavéréd'effetsgravespourlesorganes<indiquertouslesorganesaffectés,s'ilssontconnus>àlasuited'expositionsrépétéesoud'uneexpositionprolongée<indiquerlavoied'expositions'ilestformellementprouvéqu'aucuneautrevoied'expositionneconduitaumêmedanger>." in text:
        phrases.add("H372: Risque avéré d'effets graves pour les organes <indiquer tous les organes affectés, s'ils sont connus> à la suite d'expositions répétées ou d'une exposition prolongée <indiquer la voie d'exposition s'il est formellement prouvé qu'aucune autre voie d'exposition ne conduit au même danger>.")
        autres = autres.replace("Risque avéré d'effets graves pour les organes <indiquer tous les organes affectés, s'ils sont connus> à la suite d'expositions répétées ou d'une exposition prolongée <indiquer la voie d'exposition s'il est formellement prouvé qu'aucune autre voie d'exposition ne conduit au même danger>.", '')

    if "Risqueprésuméd'effetsgravespourlesorganes<ouindiquertouslesorganesaffectés,s'ilssontconnus>àlasuited'expositionsrépétéesoud'uneexpositionprolongée<indiquerlavoied'expositions'ilestformellementprouvéqu'aucuneautrevoied'expositionneconduitaumêmedanger>." in text:
        phrases.add("H373: Risque présumé d'effets graves pour les organes <ou indiquer tous les organes affectés, s'ils sont connus> à la suite d'expositions répétées ou d'une exposition prolongée <indiquer la voie d'exposition s'il est formellement prouvé qu'aucune autre voie d'exposition ne conduit au même danger>.")
        autres = autres.replace("Risque présumé d'effets graves pour les organes <ou indiquer tous les organes affectés, s'ils sont connus> à la suite d'expositions répétées ou d'une exposition prolongée <indiquer la voie d'exposition s'il est formellement prouvé qu'aucune autre voie d'exposition ne conduit au même danger>.", '')

    if "Trèstoxiquepourlesorganismesaquatiques." in text:
        phrases.add("H400: Très toxique pour les organismes aquatiques.")
        autres = autres.replace("Très toxique pour les organismes aquatiques.", '')

    if "Trèstoxiquepourlesorganismesaquatiques,entraînedeseffetsnéfastesàlongterme." in text:
        phrases.add("H410: Très toxique pour les organismes aquatiques, entraîne des effets néfastes à long terme.")
        autres = autres.replace("Très toxique pour les organismes aquatiques, entraîne des effets néfastes à long terme.", '')

    if "Toxiquepourlesorganismesaquatiques,entraînedeseffetsnéfastesàlongterme." in text:
        phrases.add("H411: Toxique pour les organismes aquatiques, entraîne des effets néfastes à long terme.")
        autres = autres.replace("Toxique pour les organismes aquatiques, entraîne des effets néfastes à long terme.", '')

    if "Nocifpourlesorganismesaquatiques,entraînedeseffetsnéfastesàlongterme." in text:
        phrases.add("H412: Nocif pour les organismes aquatiques, entraîne des effets néfastes à long terme.")
        autres = autres.replace("Nocif pour les organismes aquatiques, entraîne des effets néfastes à long terme.", '')

    if "Peutêtrenocifàlongtermepourlesorganismesaquatiques." in text:
        phrases.add("H413: Peut être nocif à long terme pour les organismes aquatiques.")
        autres = autres.replace("Peut être nocif à long terme pour les organismes aquatiques.", '')

    if "Informationsadditionnelles" in text:
        phrases.add("H413: Informations additionnelles")
        autres = autres.replace("Informations additionnelles", '')

    if "Propriétésphysiques" in text:
        phrases.add("H413: Propriétés physiques")
        autres = autres.replace("Propriétés physiques", '')

    if "Explosifàl'étatsec." in text:
        phrases.add("EUH 001: Explosif à l'état sec.")
        autres = autres.replace("Explosif à l'état sec.", '')

    if "Réagitviolemmentaucontactdel'eau." in text:
        phrases.add("EUH 014: Réagit violemment au contact de l'eau.")
        autres = autres.replace("Réagit violemment au contact de l'eau.", '')

    if "Lorsdel'utilisation,formationpossibledemélangevapeur-airinflammable/explosif." in text:
        phrases.add("EUH 018: Lors de l'utilisation, formation possible de mélange vapeur-air inflammable/explosif.")
        autres = autres.replace("Lors de l'utilisation, formation possible de mélange vapeur-air inflammable/explosif.", '')

    if "Peutformerdesperoxydesexplosifs." in text:
        phrases.add("EUH 019: Peut former des peroxydes explosifs.")
        autres = autres.replace("Peut former des peroxydes explosifs.", '')

    if "Risqued'explosionsichaufféenambianceconfinée." in text:
        phrases.add("EUH 044: Risque d'explosion si chauffé en ambiance confinée.")
        autres = autres.replace("Risque d'explosion si chauffé en ambiance confinée.", '')

    if "Propriétéssanitaires" in text:
        phrases.add("EUH 044: Propriétés sanitaires")
        autres = autres.replace("Propriétés sanitaires", '')

    if "Aucontactdel'eau,dégagedesgaztoxiques." in text:
        phrases.add("EUH 029: Au contact de l'eau, dégage des gaz toxiques.")
        autres = autres.replace("Au contact de l'eau, dégage des gaz toxiques.", '')

    if "Aucontactd'unacide,dégageungaztoxique." in text:
        phrases.add("EUH 031: Au contact d'un acide, dégage un gaz toxique.")
        autres = autres.replace("Au contact d'un acide, dégage un gaz toxique.", '')

    if "Aucontactd'unacide,dégageungaztrèstoxique." in text:
        phrases.add("EUH 032: Au contact d'un acide, dégage un gaz très toxique.")
        autres = autres.replace("Au contact d'un acide, dégage un gaz très toxique.", '')

    if "L'expositionrépétéepeutprovoquerdessèchementougerçuresdelapeau." in text:
        phrases.add("EUH 066: L'exposition répétée peut provoquer dessèchement ou gerçures de la peau.")
        autres = autres.replace("L'exposition répétée peut provoquer dessèchement ou gerçures de la peau.", '')

    if "Toxiqueparcontactoculaire." in text:
        phrases.add("EUH 070: Toxique par contact oculaire.")
        autres = autres.replace("Toxique par contact oculaire.", '')

    if "Corrosifpourlesvoiesrespiratoires." in text:
        phrases.add("EUH 071: Corrosif pour les voies respiratoires.")
        autres = autres.replace("Corrosif pour les voies respiratoires.", '')

    if "Propriétésenvironnementales" in text:
        phrases.add("EUH 071: Propriétés environnementales")
        autres = autres.replace("Propriétés environnementales", '')

    if "Élémentsadditionnelssurlesétiquettes/informationssurcertainessubstancesetpréparations" in text:
        phrases.add("EUH 071: Éléments additionnels sur les étiquettes/informations sur certaines substances et préparations")
        autres = autres.replace("Éléments additionnels sur les étiquettes/informations sur certaines substances et préparations", '')

    if "Contientduplomb.Nepasutilisersurlesobjetssusceptiblesd'êtremâchésousucéspardesenfants." in text:
        phrases.add("EUH 201: Contient du plomb. Ne pas utiliser sur les objets susceptibles d'être mâchés ou sucés par des enfants.")
        autres = autres.replace("Contient du plomb. Ne pas utiliser sur les objets susceptibles d'être mâchés ou sucés par des enfants.", '')

    if "Attention!Contientduplomb." in text:
        phrases.add("EUH 201A: Attention! Contient du plomb.")
        autres = autres.replace("Attention! Contient du plomb.", '')

    if "Cyanoacrylate.Danger.Colleàlapeauetauxyeuxenquelquessecondes.Àconserverhorsdeportéedesenfants." in text:
        phrases.add("EUH 202: Cyanoacrylate. Danger. Colle à la peau et aux yeux en quelques secondes. À conserver hors de portée des enfants.")
        autres = autres.replace("Cyanoacrylate. Danger. Colle à la peau et aux yeux en quelques secondes. À conserver hors de portée des enfants.", '')

    if "Contientduchrome(VI).Peutproduireuneréactionallergique." in text:
        phrases.add("EUH 203: Contient du chrome (VI). Peut produire une réaction allergique.")
        autres = autres.replace("Contient du chrome (VI). Peut produire une réaction allergique.", '')

    if "Contientdesisocyanates.Peutproduireuneréactionallergique." in text:
        phrases.add("EUH 204: Contient des isocyanates. Peut produire une réaction allergique.")
        autres = autres.replace("Contient des isocyanates. Peut produire une réaction allergique.", '')

    if "Contientdescomposésépoxydiques.Peutproduireuneréactionallergique." in text:
        phrases.add("EUH 205: Contient des composés époxydiques. Peut produire une réaction allergique.")
        autres = autres.replace("Contient des composés époxydiques. Peut produire une réaction allergique.", '')

    if "Attention!Nepasutiliserencombinaisonavecd'autresproduits.Peutlibérerdesgazdangereux(chlore)." in text:
        phrases.add("EUH 206: Attention! Ne pas utiliser en combinaison avec d'autres produits. Peut libérer des gaz dangereux (chlore).")
        autres = autres.replace("Attention! Ne pas utiliser en combinaison avec d'autres produits. Peut libérer des gaz dangereux (chlore).", '')

    if "Attention!Contientducadmium.Desfuméesdangereusessedéveloppentpendantl'utilisation.Voirlesinformationsfourniesparlefabricant.Respectezlesconsignesdesécurité." in text:
        phrases.add("EUH 207: Attention! Contient du cadmium. Des fumées dangereuses se développent pendant l'utilisation. Voir les informations fournies par le fabricant. Respectez les consignes de sécurité.")
        autres = autres.replace("Attention! Contient du cadmium. Des fumées dangereuses se développent pendant l'utilisation. Voir les informations fournies par le fabricant. Respectez les consignes de sécurité.", '')

    if "Contient .Peutproduireuneréactionallergique." in text:
        phrases.add("EUH 208: Contient . Peut produire une réaction allergique.")
        autres = autres.replace("Contient . Peut produire une réaction allergique.", '')

    if "Peutdevenirfacilementinflammableencoursd'utilisation." in text:
        phrases.add("EUH 209: Peut devenir facilement inflammable en cours d'utilisation.")
        autres = autres.replace("Peut devenir facilement inflammable en cours d'utilisation.", '')

    if "Peutdevenirinflammableencoursd'utilisation." in text:
        phrases.add("EUH 209A: Peut devenir inflammable en cours d'utilisation.")
        autres = autres.replace("Peut devenir inflammable en cours d'utilisation.", '')

    if "Fichededonnéesdesécuritédisponiblesurdemande." in text:
        phrases.add("EUH 210: Fiche de données de sécurité disponible sur demande.")
        autres = autres.replace("Fiche de données de sécurité disponible sur demande.", '')

    if "Respectezlesinstructionsd'utilisationpouréviterlesrisquespourlasantéhumaineetl'environnement." in text:
        phrases.add("EUH 401: Respectez les instructions d'utilisation pour éviter les risques pour la santé humaine et l'environnement.")
        autres = autres.replace("Respectez les instructions d'utilisation pour éviter les risques pour la santé humaine et l'environnement.", '')

    if "Encasdeconsultationd’unmédecin,garderàdispositionlerécipientoul’étiquette." in text:
        phrases.add("P101: En cas de consultation d’un médecin, garder à disposition le récipient ou l’étiquette.")
        autres = autres.replace("En cas de consultation d’un médecin, garder à disposition le récipient ou l’étiquette.", '')

    if "Tenirhorsdeportéedesenfants." in text:
        phrases.add("P102: Tenir hors de portée des enfants.")
        autres = autres.replace("Tenir hors de portée des enfants.", '')

    if "Lirel’étiquetteavantutilisation." in text:
        phrases.add("P103: Lire l’étiquette avant utilisation.")
        autres = autres.replace("Lire l’étiquette avant utilisation.", '')

    if "Seprocurerlesinstructionsspécialesavantutilisation." in text:
        phrases.add("P201: Se procurer les instructions spéciales avant utilisation.")
        autres = autres.replace("Se procurer les instructions spéciales avant utilisation.", '')

    if "Nepasmanipuleravantd’avoirluetcompristouteslesdispositionsdesécurité." in text:
        phrases.add("P202: Ne pas manipuler avant d’avoir lu et compris toutes les dispositions de sécurité.")
        autres = autres.replace("Ne pas manipuler avant d’avoir lu et compris toutes les dispositions de sécurité.", '')

    if "Teniràl’écartdelachaleur,dessurfaceschaudes,desétincelles,desflammesnuesetdetouteautresourced’inflammation.Nepasfumer." in text:
        phrases.add("P210: Tenir à l’écart de la chaleur, des surfaces chaudes, des étincelles, des flammes nues et de toute autre source d’inflammation. Ne pas fumer.")
        autres = autres.replace("Tenir à l’écart de la chaleur, des surfaces chaudes, des étincelles, des flammes nues et de toute autre source d’inflammation. Ne pas fumer.", '')

    if "Nepasvaporisersuruneflammenueousurtouteautresourced’ignition." in text:
        phrases.add("P211: Ne pas vaporiser sur une flamme nue ou sur toute autre source d’ignition.")
        autres = autres.replace("Ne pas vaporiser sur une flamme nue ou sur toute autre source d’ignition.", '')

    if "Teniràl'écartdesvêtementsetd'autresmatièrescombustibles." in text:
        phrases.add("P220: Tenir à l'écart des vêtements et d'autres matières combustibles.")
        autres = autres.replace("Tenir à l'écart des vêtements et d'autres matières combustibles.", '')

    if "Nepaslaisseraucontactdel’air." in text:
        phrases.add("P222: Ne pas laisser au contact de l’air.")
        autres = autres.replace("Ne pas laisser au contact de l’air.", '')

    if "Évitertoutcontactavecl’eau." in text:
        phrases.add("P223: Éviter tout contact avec l’eau.")
        autres = autres.replace("Éviter tout contact avec l’eau.", '')

    if "Maintenirhumidifiéavec..." in text:
        phrases.add("P230: Maintenir humidifié avec...")
        autres = autres.replace("Maintenir humidifié avec...", '')

    if "Manipuleretstockerlecontenusousgazinerte/…" in text:
        phrases.add("P231: Manipuler et stocker le contenu sous gaz inerte/…")
        autres = autres.replace("Manipuler et stocker le contenu sous gaz inerte/…", '')

    if "Protégerdel’humidité." in text:
        phrases.add("P232: Protéger de l’humidité.")
        autres = autres.replace("Protéger de l’humidité.", '')

    if "Maintenirlerécipientfermédemanièreétanche." in text:
        phrases.add("P233: Maintenir le récipient fermé de manière étanche.")
        autres = autres.replace("Maintenir le récipient fermé de manière étanche.", '')

    if "Conserveruniquementdansl'emballaged'origine." in text:
        phrases.add("P234: Conserver uniquement dans l'emballage d'origine.")
        autres = autres.replace("Conserver uniquement dans l'emballage d'origine.", '')

    if "Teniraufrais." in text:
        phrases.add("P235: Tenir au frais.")
        autres = autres.replace("Tenir au frais.", '')

    if "Miseàlaterreetliaisonéquipotentielledurécipientetdumatérielderéception." in text:
        phrases.add("P240: Mise à la terre et liaison équipotentielle du récipient et du matériel de réception.")
        autres = autres.replace("Mise à la terre et liaison équipotentielle du récipient et du matériel de réception.", '')

    if "Utiliserdumatériel[électrique/deventilation/d'éclairage/…]antidéflagrant." in text:
        phrases.add("P241: Utiliser du matériel [électrique/de ventilation/d'éclairage/…] antidéflagrant.")
        autres = autres.replace("Utiliser du matériel [électrique/de ventilation/d'éclairage/…] antidéflagrant.", '')

    if "Utiliserdesoutilsneproduisantpasd'étincelles." in text:
        phrases.add("P242: Utiliser des outils ne produisant pas d'étincelles.")
        autres = autres.replace("Utiliser des outils ne produisant pas d'étincelles.", '')

    if "Prendredesmesuresdeprécautioncontrelesdéchargesélectrostatiques." in text:
        phrases.add("P243: Prendre des mesures de précaution contre les décharges électrostatiques.")
        autres = autres.replace("Prendre des mesures de précaution contre les décharges électrostatiques.", '')

    if "Nihuile,nigraissesurlesrobinetsetraccords." in text:
        phrases.add("P244: Ni huile, ni graisse sur les robinets et raccords.")
        autres = autres.replace("Ni huile, ni graisse sur les robinets et raccords.", '')

    if "Éviterlesabrasions/leschocs/lesfrottements/…." in text:
        phrases.add("P250: Éviter les abrasions/les chocs/les frottements/… .")
        autres = autres.replace("Éviter les abrasions/les chocs/les frottements/… .", '')

    if "Nepasperforer,nibrûler,mêmeaprèsusage." in text:
        phrases.add("P251: Ne pas perforer, ni brûler, même après usage.")
        autres = autres.replace("Ne pas perforer, ni brûler, même après usage.", '')

    if "Nepasrespirerlespoussières/fumées/gaz/brouillards/vapeurs/aérosols." in text:
        phrases.add("P260: Ne pas respirer les poussières/fumées/gaz/brouillards/vapeurs/aérosols.")
        autres = autres.replace("Ne pas respirer les poussières/fumées/gaz/brouillards/vapeurs/aérosols.", '')

    if "Éviterderespirerlespoussières/fumées/gaz/brouillards/vapeurs/aérosols." in text:
        phrases.add("P261: Éviter de respirer les poussières/fumées/gaz/brouillards/vapeurs/aérosols.")
        autres = autres.replace("Éviter de respirer les poussières/fumées/gaz/brouillards/vapeurs/aérosols.", '')

    if "Évitertoutcontactaveclesyeux,lapeauoulesvêtements." in text:
        phrases.add("P262: Éviter tout contact avec les yeux, la peau ou les vêtements.")
        autres = autres.replace("Éviter tout contact avec les yeux, la peau ou les vêtements.", '')

    if "Évitertoutcontactaveclasubstanceaucoursdelagrossesseetpendantl'allaitement." in text:
        phrases.add("P263: Éviter tout contact avec la substance au cours de la grossesse et pendant l'allaitement.")
        autres = autres.replace("Éviter tout contact avec la substance au cours de la grossesse et pendant l'allaitement.", '')

    if "Laver…soigneusementaprèsmanipulation." in text:
        phrases.add("P264: Laver … soigneusement après manipulation.")
        autres = autres.replace("Laver … soigneusement après manipulation.", '')

    if "Nepasmanger,boireoufumerenmanipulantceproduit." in text:
        phrases.add("P270: Ne pas manger, boire ou fumer en manipulant ce produit.")
        autres = autres.replace("Ne pas manger, boire ou fumer en manipulant ce produit.", '')

    if "Utiliserseulementenpleinairoudansunendroitbienventilé." in text:
        phrases.add("P271: Utiliser seulement en plein air ou dans un endroit bien ventilé.")
        autres = autres.replace("Utiliser seulement en plein air ou dans un endroit bien ventilé.", '')

    if "Lesvêtementsdetravailcontaminésnedevraientpassortirdulieudetravail." in text:
        phrases.add("P272: Les vêtements de travail contaminés ne devraient pas sortir du lieu de travail.")
        autres = autres.replace("Les vêtements de travail contaminés ne devraient pas sortir du lieu de travail.", '')

    if "Éviterlerejetdansl’environnement." in text:
        phrases.add("P273: Éviter le rejet dans l’environnement.")
        autres = autres.replace("Éviter le rejet dans l’environnement.", '')

    if "Porterdesgantsdeprotection/desvêtementsdeprotection/unéquipementdeprotectiondesyeux/duvisage." in text:
        phrases.add("P280: Porter des gants de protection/des vêtements de protection/un équipement de protection des yeux/du visage.")
    if "desvêtementsdeprotection" in text:
        phrases.add("P280: Porter des gants de protection/des vêtements de protection/un équipement de protection des yeux/du visage.")
    if "unéquipementdeprotectiondesyeux" in text:
        phrases.add("P280: Porter des gants de protection/des vêtements de protection/un équipement de protection des yeux/du visage.")
    if "unéquipementdeprotectionduvisage." in text:
        phrases.add("P280: Porter des gants de protection/des vêtements de protection/un équipement de protection des yeux/du visage.")
    if "Porterdesgantsdeprotection" in text:
        phrases.add("P280: Porter des gants de protection/des vêtements de protection/un équipement de protection des yeux/du visage.")
        autres = autres.replace("Porter des gants de protection/des vêtements de protection/un équipement de protection des yeux/du visage.", '')

    if "Porterdesgantsisolantscontrelefroidetunéquipementdeprotectionduvisageoudesyeux." in text:
        phrases.add("P282: Porter des gants isolants contre le froid et un équipement de protection du visage ou des yeux.")
        autres = autres.replace("Porter des gants isolants contre le froid et un équipement de protection du visage ou des yeux.", '')

    if "Porterdesvêtementsrésistantaufeuouàretarddeflamme." in text:
        phrases.add("P283: Porter des vêtements résistant au feu ou à retard de flamme.")
        autres = autres.replace("Porter des vêtements résistant au feu ou à retard de flamme.", '')

    if "[Lorsquelaventilationdulocalestinsuffisante]porterunéquipementdeprotectionrespiratoire." in text:
        phrases.add("P284: [Lorsque la ventilation du local est insuffisante] porter un équipement de protection respiratoire.")
        autres = autres.replace("[Lorsque la ventilation du local est insuffisante] porter un équipement de protection respiratoire.", '')

    if "Manipuleretstockerlecontenusousgazinerte/…Protégerdel'humidité." in text:
        phrases.add("P231 + P232: Manipuler et stocker le contenu sous gaz inerte/… Protéger de l'humidité.")
        autres = autres.replace("Manipuler et stocker le contenu sous gaz inerte/… Protéger de l'humidité.", '')

    if "ENCASD’INGESTION:" in text:
        phrases.add("P301: EN CAS D’INGESTION:")
        autres = autres.replace("EN CAS D’INGESTION:", '')

    if "ENCASDECONTACTAVECLAPEAU:" in text:
        phrases.add("P302: EN CAS DE CONTACT AVEC LA PEAU:")
        autres = autres.replace("EN CAS DE CONTACT AVEC LA PEAU:", '')

    if "ENCASDECONTACTAVECLAPEAU(oulescheveux):" in text:
        phrases.add("P303: EN CAS DE CONTACT AVEC LA PEAU (ou les cheveux):")
        autres = autres.replace("EN CAS DE CONTACT AVEC LA PEAU (ou les cheveux):", '')

    if "ENCASD’INHALATION:" in text:
        phrases.add("P304: EN CAS D’INHALATION:")
        autres = autres.replace("EN CAS D’INHALATION:", '')

    if "ENCASDECONTACTAVECLESYEUX:" in text:
        phrases.add("P305: EN CAS DE CONTACT AVEC LES YEUX:")
        autres = autres.replace("EN CAS DE CONTACT AVEC LES YEUX:", '')

    if "ENCASDECONTACTAVECLESVÊTEMENTS:" in text:
        phrases.add("P306: EN CAS DE CONTACT AVEC LES VÊTEMENTS:")
        autres = autres.replace("EN CAS DE CONTACT AVEC LES VÊTEMENTS:", '')

    if "ENCASd’expositionprouvéeoususpectée:" in text:
        phrases.add("P308: EN CAS d’exposition prouvée ou suspectée:")
        autres = autres.replace("EN CAS d’exposition prouvée ou suspectée:", '')

    if "AppelerimmédiatementunCENTREANTIPOISON/unmédecin/…" in text:
        phrases.add("P310: Appeler immédiatement un CENTRE ANTIPOISON/un médecin/…")
        autres = autres.replace("Appeler immédiatement un CENTRE ANTIPOISON/un médecin/…", '')

    if "AppelerunCENTREANTIPOISON/unmédecin/…" in text:
        phrases.add("P311: Appeler un CENTRE ANTIPOISON/un médecin/…")
        autres = autres.replace("Appeler un CENTRE ANTIPOISON/un médecin/…", '')

    if "AppelerunCENTREANTIPOISON/unmédecin/…/encasdemalaise." in text:
        phrases.add("P312: Appeler un CENTRE ANTIPOISON/un médecin/…/ en cas de malaise.")
        autres = autres.replace("Appeler un CENTRE ANTIPOISON/un médecin/…/ en cas de malaise.", '')

    if "Consulterunmédecin." in text:
        phrases.add("P313: Consulter un médecin.")
        autres = autres.replace("Consulter un médecin.", '')

    if "Consulterunmédecinencasdemalaise." in text:
        phrases.add("P314: Consulter un médecin en cas de malaise.")
        autres = autres.replace("Consulter un médecin en cas de malaise.", '')

    if "Consulterimmédiatementunmédecin." in text:
        phrases.add("P315: Consulter immédiatement un médecin.")
        autres = autres.replace("Consulter immédiatement un médecin.", '')

    if "Untraitementspécifiqueesturgent." in text:
        phrases.add("P320: Un traitement spécifique est urgent (voir ... sur cette étiquette).")
        autres = autres.replace("Un traitement spécifique est urgent (voir ... sur cette étiquette).", '')

    if "Traitementspécifique." in text:
        phrases.add("P321: Traitement spécifique (voir ... sur cette étiquette).")
        autres = autres.replace("Traitement spécifique (voir ... sur cette étiquette).", '')

    if "Rincerlabouche." in text:
        phrases.add("P330: Rincer la bouche.")
        autres = autres.replace("Rincer la bouche.", '')

    if "NEPASfairevomir." in text:
        phrases.add("P331: NE PAS faire vomir.")
        autres = autres.replace("NE PAS faire vomir.", '')

    if "Encasd’irritationcutanée:" in text:
        phrases.add("P332: En cas d’irritation cutanée:")
        autres = autres.replace("En cas d’irritation cutanée:", '')

    if "Encasd’irritationoud’éruptioncutanée:" in text:
        phrases.add("P333: En cas d’irritation ou d’éruption cutanée:")
        autres = autres.replace("En cas d’irritation ou d’éruption cutanée:", '')

    if "Rinceràl'eaufraîche[ouposerunecompressehumide]." in text:
        phrases.add("P334: Rincer à l'eau fraîche [ou poser une compresse humide].")
        autres = autres.replace("Rincer à l'eau fraîche [ou poser une compresse humide].", '')

    if "Enleveravecprécautionlesparticulesdéposéessurlapeau." in text:
        phrases.add("P335: Enlever avec précaution les particules déposées sur la peau.")
        autres = autres.replace("Enlever avec précaution les particules déposées sur la peau.", '')

    if "Dégelerlespartiesgeléesavecdel’eautiède.Nepasfrotterleszonestouchées." in text:
        phrases.add("P336: Dégeler les parties gelées avec de l’eau tiède. Ne pas frotter les zones touchées.")
        autres = autres.replace("Dégeler les parties gelées avec de l’eau tiède. Ne pas frotter les zones touchées.", '')

    if "Sil’irritationoculairepersiste:" in text:
        phrases.add("P337: Si l’irritation oculaire persiste:")
        autres = autres.replace("Si l’irritation oculaire persiste:", '')

    if "Enleverleslentillesdecontactsilavictimeenporteetsiellespeuventêtrefacilementenlevées.Continueràrincer." in text:
        phrases.add("P338: Enlever les lentilles de contact si la victime en porte et si elles peuvent être facilement enlevées. Continuer à rincer.")
        autres = autres.replace("Enlever les lentilles de contact si la victime en porte et si elles peuvent être facilement enlevées. Continuer à rincer.", '')

    if "Transporterlapersonneàl’extérieuretlamaintenirdansunepositionoùellepeutconfortablementrespirer." in text:
        phrases.add("P340: Transporter la personne à l’extérieur et la maintenir dans une position où elle peut confortablement respirer.")
        autres = autres.replace("Transporter la personne à l’extérieur et la maintenir dans une position où elle peut confortablement respirer.", '')

    if "Encasdesymptômesrespiratoires:" in text:
        phrases.add("P342: En cas de symptômes respiratoires:")
        autres = autres.replace("En cas de symptômes respiratoires:", '')

    if "Rinceravecprécautionàl’eaupendantplusieursminutes." in text:
        phrases.add("P351: Rincer avec précaution à l’eau pendant plusieurs minutes.")
        autres = autres.replace("Rincer avec précaution à l’eau pendant plusieurs minutes.", '')

    if "Laverabondammentàl’eau/…" in text:
        phrases.add("P352: Laver abondamment à l’eau/…")
        autres = autres.replace("Laver abondamment à l’eau/…", '')

    if "Rincerlapeauàl'eau[ousedoucher]." in text:
        phrases.add("P353: Rincer la peau à l'eau [ou se doucher].")
        autres = autres.replace("Rincer la peau à l'eau [ou se doucher].", '')

    if "Rincerimmédiatementetabondammentavecdel’eaulesvêtementscontaminésetlapeauavantdelesenlever." in text:
        phrases.add("P360: Rincer immédiatement et abondamment avec de l’eau les vêtements contaminés et la peau avant de les enlever.")
        autres = autres.replace("Rincer immédiatement et abondamment avec de l’eau les vêtements contaminés et la peau avant de les enlever.", '')

    if "Enleverimmédiatementtouslesvêtementscontaminés." in text:
        phrases.add("P361: Enlever immédiatement tous les vêtements contaminés.")
        autres = autres.replace("Enlever immédiatement tous les vêtements contaminés.", '')

    if "Enleverlesvêtementscontaminés." in text:
        phrases.add("P362: Enlever les vêtements contaminés.")
        autres = autres.replace("Enlever les vêtements contaminés.", '')

    if "Laverlesvêtementscontaminésavantréutilisation." in text:
        phrases.add("P363: Laver les vêtements contaminés avant réutilisation.")
        autres = autres.replace("Laver les vêtements contaminés avant réutilisation.", '')

    if "Encasd’incendie:" in text:
        phrases.add("P370: En cas d’incendie:")
        autres = autres.replace("En cas d’incendie:", '')

    if "Encasd’incendieimportantets’ils’agitdegrandesquantités:" in text:
        phrases.add("P371: En cas d’incendie important et s’il s’agit de grandes quantités:")
        autres = autres.replace("En cas d’incendie important et s’il s’agit de grandes quantités:", '')

    if "Risqued'explosion." in text:
        phrases.add("P372: Risque d'explosion.")
        autres = autres.replace("Risque d'explosion.", '')

    if "NEPAScombattrel’incendielorsquelefeuatteintlesexplosifs." in text:
        phrases.add("P373: NE PAS combattre l’incendie lorsque le feu atteint les explosifs.")
        autres = autres.replace("NE PAS combattre l’incendie lorsque le feu atteint les explosifs.", '')

    if "Combattrel’incendieàdistanceàcausedurisqued’explosion." in text:
        phrases.add("P375: Combattre l’incendie à distance à cause du risque d’explosion.")
        autres = autres.replace("Combattre l’incendie à distance à cause du risque d’explosion.", '')

    if "Obturerlafuitesicelapeutsefairesansdanger." in text:
        phrases.add("P376: Obturer la fuite si cela peut se faire sans danger.")
        autres = autres.replace("Obturer la fuite si cela peut se faire sans danger.", '')

    if "Fuitedegazenflammé:" in text:
        phrases.add("P377: Fuite de gaz enflammé:")
        autres = autres.replace("Fuite de gaz enflammé:", '')

    if "Nepaséteindresilafuitenepeutpasêtrearrêtéesansdanger." in text:
        phrases.add("P377: Ne pas éteindre si la fuite ne peut pas être arrêtée sans danger.")
        autres = autres.replace("Ne pas éteindre si la fuite ne peut pas être arrêtée sans danger.", '')

    if "Utiliser…pourl’extinction." in text:
        phrases.add("P378: Utiliser… pour l’extinction.")
        autres = autres.replace("Utiliser… pour l’extinction.", '')

    if "Évacuerlazone." in text:
        phrases.add("P380: Évacuer la zone.")
        autres = autres.replace("Évacuer la zone.", '')

    if "Encasdefuite,éliminertouteslessourcesd'ignition." in text:
        phrases.add("P381: En cas de fuite, éliminer toutes les sources d'ignition.")
        autres = autres.replace("En cas de fuite, éliminer toutes les sources d'ignition.", '')

    if "Absorbertoutesubstancerépanduepouréviterqu’elleattaquelesmatériauxenvironnants." in text:
        phrases.add("P390: Absorber toute substance répandue pour éviter qu’elle attaque les matériaux environnants.")
        autres = autres.replace("Absorber toute substance répandue pour éviter qu’elle attaque les matériaux environnants.", '')

    if "Recueillirleproduitrépandu." in text:
        phrases.add("P391: Recueillir le produit répandu.")
        autres = autres.replace("Recueillir le produit répandu.", '')

    if "ENCASD’INGESTION:AppelerimmédiatementunCENTREANTIPOISON/unmédecin/…" in text:
        phrases.add("P301 + P310: EN CAS D’INGESTION: Appeler immédiatement un CENTRE ANTIPOISON/un médecin/…")
        autres = autres.replace("EN CAS D’INGESTION: Appeler immédiatement un CENTRE ANTIPOISON/un médecin/…", '')

    if "ENCASD'INGESTION:AppelerunCENTREANTIPOISON/unmédecin/…/encasdemalaise." in text:
        phrases.add("P301 + P312: EN CAS D'INGESTION: Appeler un CENTRE ANTIPOISON/un médecin/…/ en cas de malaise.")
        autres = autres.replace("EN CAS D'INGESTION: Appeler un CENTRE ANTIPOISON/un médecin/…/ en cas de malaise.", '')

    if "ENCASD’INGESTION:rincerlabouche.NEPASfairevomir." in text:
        phrases.add("P301 + P330 + P331: EN CAS D’INGESTION: rincer la bouche. NE PAS faire vomir.")
        autres = autres.replace("EN CAS D’INGESTION: rincer la bouche. NE PAS faire vomir.", '')

    if "ENCASDECONTACTAVECLAPEAU:Rinceràl'eaufraîcheouposerunecompressehumide." in text:
        phrases.add("P302 + P334: EN CAS DE CONTACT AVEC LA PEAU: Rincer à l'eau fraîche ou poser une compresse humide.")
        autres = autres.replace("EN CAS DE CONTACT AVEC LA PEAU: Rincer à l'eau fraîche ou poser une compresse humide.", '')

    if "ENCASODECONTACTOCONLAPIEL:Laverabondammentàl’eau/…" in text:
        phrases.add("P302 + P352: EN CASO DE CONTACTO CON LA PIEL: Laver abondamment à l’eau/…")
        autres = autres.replace("EN CASO DE CONTACTO CON LA PIEL: Laver abondamment à l’eau/…", '')

    if "ENCASDECONTACTAVECLAPEAU(oulescheveux):Enleverimmédiatementtouslesvêtementscontaminés.Rincerlapeauàl'eau[ousedoucher]." in text:
        phrases.add("P303 + P361 + P353: EN CAS DE CONTACT AVEC LA PEAU (ou les cheveux): Enlever immédiatement tous les vêtements contaminés. Rincer la peau à l'eau [ou se doucher].")
        autres = autres.replace("EN CAS DE CONTACT AVEC LA PEAU (ou les cheveux): Enlever immédiatement tous les vêtements contaminés. Rincer la peau à l'eau [ou se doucher].", '')

    if "ENCASD’INHALATION:transporterlapersonneàl’extérieuretlamaintenirdansunepositionoùellepeutconfortablementrespirer." in text:
        phrases.add("P304 + P340: EN CAS D’INHALATION: transporter la personne à l’extérieur et la maintenir dans une position où elle peut confortablement respirer.")
        autres = autres.replace("EN CAS D’INHALATION: transporter la personne à l’extérieur et la maintenir dans une position où elle peut confortablement respirer.", '')

    if "ENCASDECONTACTAVECLESYEUX:rinceravecprécautionàl’eaupendantplusieursminutes.Enleverleslentillesdecontactsilavictimeenporteetsiellespeuventêtrefacilementenlevées.Continueràrincer." in text:
        phrases.add("P305 + P351 + P338: EN CAS DE CONTACT AVEC LES YEUX: rincer avec précaution à l’eau pendant plusieurs minutes. Enlever les lentilles de contact si la victime en porte et si elles peuvent être facilement enlevées. Continuer à rincer.")
        autres = autres.replace("EN CAS DE CONTACT AVEC LES YEUX: rincer avec précaution à l’eau pendant plusieurs minutes. Enlever les lentilles de contact si la victime en porte et si elles peuvent être facilement enlevées. Continuer à rincer.", '')

    if "ENCASDECONTACTAVECLESVÊTEMENTS:rincerimmédiatementetabondammentavecdel’eaulesvêtementscontaminésetlapeauavantdelesenlever." in text:
        phrases.add("P306 + P360: EN CAS DE CONTACT AVEC LES VÊTEMENTS: rincer immédiatement et abondamment avec de l’eau les vêtements contaminés et la peau avant de les enlever.")
        autres = autres.replace("EN CAS DE CONTACT AVEC LES VÊTEMENTS: rincer immédiatement et abondamment avec de l’eau les vêtements contaminés et la peau avant de les enlever.", '')

    if "ENCASd’expositionprouvéeoususpectée:consulterunmédecin." in text:
        phrases.add("P308 + P313: EN CAS d’exposition prouvée ou suspectée: consulter un médecin.")
        autres = autres.replace("EN CAS d’exposition prouvée ou suspectée: consulter un médecin.", '')

    if "Encasd’irritationcutanée:consulterunmédecin." in text:
        phrases.add("P332 + P313: En cas d’irritation cutanée: consulter un médecin.")
        autres = autres.replace("En cas d’irritation cutanée: consulter un médecin.", '')

    if "Encasd’irritationoud'éruptioncutanée:consulterunmédecin." in text:
        phrases.add("P333 + P313: En cas d’irritation ou d'éruption cutanée: consulter un médecin.")
        autres = autres.replace("En cas d’irritation ou d'éruption cutanée: consulter un médecin.", '')

    if "Sil’irritationoculairepersiste:consulterunmédecin." in text:
        phrases.add("P337 + P313: Si l’irritation oculaire persiste: consulter un médecin.")
        autres = autres.replace("Si l’irritation oculaire persiste: consulter un médecin.", '')

    if "Encasdesymptômesrespiratoires:AppelerunCENTREANTIPOISON/unmédecin/…" in text:
        phrases.add("P342 + P311: En cas de symptômes respiratoires: Appeler un CENTRE ANTIPOISON/un médecin/…")
        autres = autres.replace("En cas de symptômes respiratoires: Appeler un CENTRE ANTIPOISON/un médecin/…", '')

    if "Encasd’incendie:obturerlafuitesicelapeutsefairesansdanger." in text:
        phrases.add("P370 + P376: En cas d’incendie: obturer la fuite si cela peut se faire sans danger.")
        autres = autres.replace("En cas d’incendie: obturer la fuite si cela peut se faire sans danger.", '')

    if "Encasd’incendie:Utiliser…pourl’extinction." in text:
        phrases.add("P370 + P378: En cas d’incendie: Utiliser… pour l’extinction.")
        autres = autres.replace("En cas d’incendie: Utiliser… pour l’extinction.", '')

    if "Encasd’incendie:évacuerlazone.Combattrel’incendieàdistanceàcausedurisqued’explosion." in text:
        phrases.add("P370 + P380 + P375: En cas d’incendie: évacuer la zone. Combattre l’incendie à distance à cause du risque d’explosion.")
        autres = autres.replace("En cas d’incendie: évacuer la zone. Combattre l’incendie à distance à cause du risque d’explosion.", '')

    if "Encasd’incendieimportantets’ils’agitdegrandesquantités:évacuerlazone.Combattrel’incendieàdistanceàcausedurisqued’explosion." in text:
        phrases.add("P371 + P380 + P375: En cas d’incendie important et s’il s’agit de grandes quantités: évacuer la zone. Combattre l’incendie à distance à cause du risque d’explosion.")
        autres = autres.replace("En cas d’incendie important et s’il s’agit de grandes quantités: évacuer la zone. Combattre l’incendie à distance à cause du risque d’explosion.", '')

    if "Stockerconformémentà…." in text:
        phrases.add("P401: Stocker conformément à… .")
        autres = autres.replace("Stocker conformément à… .", '')

    if "Stockerdansunendroitsec." in text:
        phrases.add("P402: Stocker dans un endroit sec.")
        autres = autres.replace("Stocker dans un endroit sec.", '')

    if "Stockerdansunendroitbienventilé." in text:
        phrases.add("P403: Stocker dans un endroit bien ventilé.")
        autres = autres.replace("Stocker dans un endroit bien ventilé.", '')

    if "Stockerdansunrécipientfermé." in text:
        phrases.add("P404: Stocker dans un récipient fermé.")
        autres = autres.replace("Stocker dans un récipient fermé.", '')

    if "Gardersousclef." in text:
        phrases.add("P405: Garder sous clef.")
        autres = autres.replace("Garder sous clef.", '')

    if "Stockerdansunrécipientrésistantàlacorrosion/…avecdoublureintérieure." in text:
        phrases.add("P406: Stocker dans un récipient résistant à la corrosion/… avec doublure intérieure.")
        autres = autres.replace("Stocker dans un récipient résistant à la corrosion/… avec doublure intérieure.", '')

    if "Maintenirunintervalled'airentrelespilesoulespalettes." in text:
        phrases.add("P407: Maintenir un intervalle d'air entre les piles ou les palettes.")
        autres = autres.replace("Maintenir un intervalle d'air entre les piles ou les palettes.", '')

    if "Protégerdurayonnementsolaire." in text:
        phrases.add("P410: Protéger du rayonnement solaire.")
        autres = autres.replace("Protéger du rayonnement solaire.", '')

    if "Stockeràunetempératurenedépassantpas...°C/...°F." in text:
        phrases.add("P411: Stocker à une température ne dépassant pas ... °C/... °F.")
        autres = autres.replace("Stocker à une température ne dépassant pas ... °C/... °F.", '')

    if "Nepasexposeràunetempératuresupérieureà50°C/122°F." in text:
        phrases.add("P412: Ne pas exposer à une température supérieure à 50 °C/122 °F.")
        autres = autres.replace("Ne pas exposer à une température supérieure à 50 °C/122 °F.", '')

    if "Stockerlesquantitésenvracdeplusde...kg/...lbàunetempératurenedépassantpas...°C/...°F." in text:
        phrases.add("P413: Stocker les quantités en vrac de plus de ... kg/... lb à une température ne dépassant pas ... °C/... °F.")
        autres = autres.replace("Stocker les quantités en vrac de plus de ... kg/... lb à une température ne dépassant pas ... °C/... °F.", '')

    if "Stockerséparément." in text:
        phrases.add("P420: Stocker séparément.")
        autres = autres.replace("Stocker séparément.", '')

    if "Stockerdansunendroitsec.Stockerdansunrécipientfermé." in text:
        phrases.add("P402 + P404: Stocker dans un endroit sec. Stocker dans un récipient fermé.")
        autres = autres.replace("Stocker dans un endroit sec. Stocker dans un récipient fermé.", '')

    if "Stockerdansunendroitbienventilé.Maintenirlerécipientfermédemanièreétanche." in text:
        phrases.add("P403 + P233: Stocker dans un endroit bien ventilé. Maintenir le récipient fermé de manière étanche.")
        autres = autres.replace("Stocker dans un endroit bien ventilé. Maintenir le récipient fermé de manière étanche.", '')

    if "Stockerdansunendroitbienventilé.Teniraufrais." in text:
        phrases.add("P403 + P235: Stocker dans un endroit bien ventilé. Tenir au frais.")
        autres = autres.replace("Stocker dans un endroit bien ventilé. Tenir au frais.", '')

    if "Protégerdurayonnementsolaire.Stockerdansunendroitbienventilé." in text:
        phrases.add("P410 + P403: Protéger du rayonnement solaire. Stocker dans un endroit bien ventilé.")
        autres = autres.replace("Protéger du rayonnement solaire. Stocker dans un endroit bien ventilé.", '')

    if "Protégerdurayonnementsolaire.Nepasexposeràunetempératuresupérieureà50°C/122°F." in text:
        phrases.add("P410 + P412: Protéger du rayonnement solaire. Ne pas exposer à une température supérieure à 50 °C/ 122 °F.")
        autres = autres.replace("Protéger du rayonnement solaire. Ne pas exposer à une température supérieure à 50 °C/ 122 °F.", '')

    if "Éliminerlecontenu/récipientdans..." in text:
        phrases.add("P501: Éliminer le contenu/récipient dans ...")
        autres = autres.replace("Éliminer le contenu/récipient dans ...", '')
    if "Eliminerlecontenuet/oulerécipientdansuncentreagrééconformémentàlaréglementationnationale" in text:
        phrases.add("P501: Éliminer le contenu et/ou le récipient dans un centre agréé conformément à la réglementation nationale.")
        autres = autres.replace("Eliminer le contenu et/ou le récipient dans un centre agréé conformément à la réglementation nationale", '')

    clp += '\n'.join(phrases)  # Ajouter un saut de ligne entre chaque phrase CLP
    clp += '.\n'  # Ajouter un point à la fin de la liste de phrases CLP
    autres = autres.replace(clp, '')

    if "CLP:" in text:
        clp = "Votre texte CLP ici..."
        autres = autres.replace(clp, '')

    return mode_d_emploi, marketing, clp, autres

import cv2
import numpy as np

def find_largest_bounding_box_and_measure(images, dpi):
    largest_area = 0
    largest_dimensions = (0, 0)  # Initialiser avec une largeur et une hauteur de 0
    image_with_largest_box = None
    for image in images:
        # Convertir l'image en nuances de gris pour simplifier le traitement
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Appliquer un seuil pour obtenir une image binaire où les zones non blanches sont noires
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

        # Trouver les contours dans l'image binaire
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Calculer le rectangle englobant pour chaque contour
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            # Vérifier si ce rectangle est le plus grand trouvé jusqu'à présent
            if area > largest_area:
                largest_area = area
                largest_dimensions = (w, h)
                image_with_largest_box = image.copy()  # Faire une copie de l'image pour dessiner le rectangle
                # Dessiner le rectangle sur l'image copiée
                cv2.rectangle(image_with_largest_box, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Convertir les dimensions du plus grand rectangle de pixels en millimètres en utilisant le DPI
    if largest_area > 0:
        width_mm = (largest_dimensions[0] * 25.4) / dpi
        height_mm = (largest_dimensions[1] * 25.4) / dpi
        return width_mm, height_mm, image_with_largest_box
    else:
        return None, None, None


def main():
    sg.theme('SystemDefault')

    layout = [
        [sg.Text("             CRF CLP DETECTOR", font=('Many Years Higher', 40), key='TETXTT',  justification='center')],
        [sg.Text("Choisir un fichier pdf:"), sg.Input(), sg.FileBrowse(file_types=(("PDF Files", "*.pdf"),)), sg.Button("Process")],
        [sg.Text("Pictogramme CLP détecté (côté de carré) :", size=(40, 1)), sg.Text("", size=(40, 1), key="-SIZE-", text_color='black')],
        [sg.Text("Volume du contenant :", size=(40, 1)), sg.Text("", size=(30, 1), key="-VOLUME-")],
        [sg.Text("Taille du Packaging :", size=(40, 1)), sg.Text("", size=(40, 1), key="-PACKSIZE-")],
        [sg.Text("Surface du pictogramme CLP / Surface du packaging :", size=(43, 1)), sg.Text("", size=(40, 1), key="-RATIO-")],
        [sg.Button("Picto CLP", key="-SHOW_IMAGE-"), sg.Button("Cadre Packaging", key="-SHOW_PACKAGING-"), sg.Button("Texte Pack", key="-SHOW_TEXT-"), sg.Multiline(default_text="CLP Phrases will be displayed here", size=(80, 20), key="-CLP_PHRASES-", disabled=True)],
        [sg.Button("Mode d'emploi"), sg.Button("Marketing"), sg.Button("CLP"), sg.Button("Autres")]
    ]

    window = sg.Window("CLP Pictogram Detector and Measurer", layout, finalize=True, resizable=True, size=(700, 500))

    pdf_text, image_to_show, image_with_largest_box = "", None, None
    pack_width_mm, pack_height_mm = 0,0
    completion=''
    response=''
    mark_text=''
    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSED:
            break
        elif event == "Process":
            pdf_path = values[0]
            if pdf_path:
                pdf_text, images, dpi = extract_text_and_images_from_pdf(pdf_path)
                volume_text = extract_volume_from_text(pdf_text)
                window["-VOLUME-"].update(volume_text)
                _, _, image_with_largest_box = find_largest_bounding_box_and_measure(images, dpi)
                pdf_texto=pdf_text
                pdf_texto = pdf_texto.replace('\n', ' ')
                volume = None
                if "Volume not found" not in volume_text:
                    try:
                        volume = float(re.search(r'\d+(\.\d+)?', volume_text).group(0)) if "L" in volume_text.upper() else float(re.search(r'\d+(\.\d+)?', volume_text).group(0))/1000
                    except Exception:
                        volume = None
                packsize_text = extract_packaging_size_from_text(pdf_text)
                window["-PACKSIZE-"].update(packsize_text)
                if "Packaging" in packsize_text :
                    packsize_text=find_smallest_bounding_box_and_measure(images, dpi)
                window["-PACKSIZE-"].update(packsize_text)
                pdf_texto=pdf_text
                pdf_texto = pdf_texto.replace('\n', ' ')
                pdf_texto=pdf_text
                pdf_texto = pdf_texto.replace('\n', ' ')
                window["-CLP_PHRASES-"].update(classify_text(pdf_texto)[2])
                for image in images:
                    size_mm, annotated_image = detect_clp_pictogram_and_measure(image, dpi)
                    if size_mm and annotated_image is not None:
                        image_to_show = resize_and_convert_image_to_bytes(annotated_image)
                        if volume is not None:
                            if ((volume < 3 and size_mm >= 16) or (volume >= 3 and size_mm >= 23)):
                                alert_text = f"✅ {size_mm:.2f} mm (CONFORME)"
                                window["-SIZE-"].update(alert_text, text_color='green')
                            else:
                                alert_text = f"⚠️ {size_mm:.2f} mm (non conforme)"
                                window["-SIZE-"].update(alert_text, text_color='red')
                        else:
                            window["-SIZE-"].update(f"{size_mm:.2f} mm", text_color='black')
                        break
                else:
                    window["-SIZE-"].update("Pictogram not detected")

            if "x" in packsize_text:  # Si les dimensions sont trouvées dans le texte
                dimensions = [int(s) for s in packsize_text.split() if s.isdigit()]
                if len(dimensions) == 2:
                    pack_width_mm, pack_height_mm = dimensions
            else:  # Si les dimensions ne sont pas trouvées dans le texte
                # Utiliser find_smallest_bounding_box_and_measure pour obtenir les dimensions
                packsize_text, _ = find_smallest_bounding_box_and_measure(images, dpi)
                if packsize_text:  # Assurez-vous que packsize_text n'est pas None
                    dimensions = [int(s) for s in packsize_text.split() if s.isdigit()]
                    if len(dimensions) == 2:
                        pack_width_mm, pack_height_mm = dimensions

            # Calculer les surfaces en mm²
            clp_surface_mm2 = size_mm * size_mm
            packaging_surface_mm2 = pack_width_mm * pack_height_mm

            # Calcul du pourcentage du pictogramme CLP par rapport à la surface du packaging
            if packaging_surface_mm2 > 0:  # Assurez-vous de ne pas diviser par zéro
                percentage = (clp_surface_mm2 / packaging_surface_mm2) * 100
                percentage_str = f"{percentage:.2f}%"
                window["-RATIO-"].update(percentage_str)
            else:
                window["-RATIO-"].update("Dimensions non trouvées ou invalide")


        elif event == "-SHOW_IMAGE-" and image_to_show:
            layout = [[sg.Image(data=image_to_show)]]
            popup_window = sg.Window("Annotated CLP Pictogram", layout, modal=True, resizable=True)
            popup_event, popup_values = popup_window.read()
            popup_window.close()


        elif event == "-SHOW_PACKAGING-" and image_with_largest_box is not None:
            # Calculer le ratio de mise à l'échelle pour conserver le ratio d'aspect
            height, width = image_with_largest_box.shape[:2]
            scale_width = 1600 / width
            scale_height = 900 / height
            scale = min(scale_width, scale_height)

            # Appliquer le redimensionnement avec le ratio conservé
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized_image = cv2.resize(image_with_largest_box, (new_width, new_height), interpolation=cv2.INTER_AREA)

            # Convertir l'image redimensionnée en bytes pour l'affichage
            img_bytes = cv2.imencode('.png', resized_image)[1].tobytes()

            layout = [[sg.Image(data=img_bytes)]]
            popup_window = sg.Window("Largest Packaging Bounding Box", layout, modal=True, resizable=True)
            popup_event, popup_values = popup_window.read()
            popup_window.close()


        elif event == "-SHOW_TEXT-":
            pdf_texto=pdf_text
            pdf_texto+="\n \n Trie le texte suivant en différents paragraphes car tout le texte a été mis à la chaîne et garde chaque mot et chaque langue et mets tout en Markdown"
            response = chat.send_message(pdf_texto)
            layout = [[sg.Multiline(response.text, size=(80, 25), disabled=True, autoscroll=True, font=("Helvetica", 12))]]
            sg.Window("Markdown Viewer", layout, finalize=True)

        elif event == "Mode d'emploi":
            pdf_texto=pdf_text
            pdf_texto+="\n \n Trie le texte suivant en différents paragraphes car tout le texte a été mis à la chaîne et garde chaque mot et chaque langue"
            response = chat.send_message(pdf_texto)
            sg.popup_scrolled(pdf_texto, title="PDF Text Content", size=(80, 25))

        elif event == "Marketing":
            pdf_texto=pdf_text
            pdf_texto = pdf_texto.replace('\n', ' ')
            sg.popup_scrolled(classify_text(pdf_texto)[1], title="Marketing", size=(80, 25))

        elif event == "CLP":
            pdf_texto=pdf_text
            pdf_texto = pdf_texto.replace('\n', ' ')
            sg.popup_scrolled(classify_text(pdf_texto)[2], title="CLP", size=(80, 25))

        elif event == "Autres":
            pdf_texto=pdf_text
            pdf_texto = pdf_texto.replace('\n', ' ')
            sg.popup_scrolled(classify_text(pdf_texto)[3], title="Autres", size=(80, 25))

    window.close()

if __name__ == "__main__":
    main()

