import json
import string
from phonemizer import phonemize
from phonemizer.separator import Separator
from tensorflow.keras.preprocessing.sequence import pad_sequences
from app.params import *
from app.utils import *
import textgrids
import numpy as np
# from g2p_en import G2p

# TODO: OK sauf les phonemize transcripts et tokenize and pad transcripts et get input sample


def get_tokenized_transcripts(phonemized_transcripts_dict, mapping_file=PATH_PHONES_MAPPING_NEW):
    """
    Convertit un dictionnaire de transcriptions phonetisées en un dictionnaire de transcriptions tokenisées

    Parameters:
    - phonemized_transcripts_dict : un dictionnaire avec les sequence_id comme clés et les listes de phonèmes comme valeurs.
    - mapping_file : path vers le fichier JSON contenant le mapping des phonèmes en tokens.
    Retour:
    - tokens_dict: Dictionnaire avec les sequence_id comme clés et les listes de tokens comme valeurs.
    """
    with open(mapping_file, 'r') as file:
        phoneme_mapping_unicode = json.load(file)

    phoneme_mapping = {key: value for key,
                       value in phoneme_mapping_unicode.items()}

    tokenized_transcriptions_dict = {}

    for sequence_id, phonemized_transcripts_array in phonemized_transcripts_dict.items():
        token_phonem_sequence = [phoneme_mapping[phonem]
                                 for phonem in phonemized_transcripts_array if phonem in phoneme_mapping]
        tokenized_transcriptions_dict[sequence_id] = token_phonem_sequence

    return tokenized_transcriptions_dict

# def get_phonems_from_tokens(tokens_list, mapping_file=PATH_PHONES_MAPPING_new):
    """
    Convertit une liste de tokens en une liste de phonèmes.
    
    Parameters:
    - tokens_list : Liste de tokens à convertir.
    - mapping_file : Chemin vers le fichier JSON contenant le mapping des phonèmes en tokens.
    
    Retour:
    - phonems_list: Liste des phonèmes correspondants aux tokens.
    """
    # Charger le mapping à partir du fichier JSON
    with open(mapping_file, 'r') as file:
        phoneme_mapping = json.load(file)

    # Convertir les séquences d'échappement Unicode en caractères réels
    phoneme_mapping = {key: value for key, value in phoneme_mapping.items()}

    # Inverser le mapping pour obtenir token -> phonème
    inverted_mapping = {int(value): key for key,
                        value in phoneme_mapping.items()}

    # Convertir les tokens en phonèmes
    phonems_list = [inverted_mapping[token]
                    for token in tokens_list if token in inverted_mapping]

    # Gérer le cas où un token n'est pas trouvé dans le mapping
    if len(phonems_list) != len(tokens_list):
        missing_tokens = [
            token for token in tokens_list if token not in inverted_mapping]
        raise ValueError(
            f"Certains tokens n'ont pas été trouvés dans le mapping: {missing_tokens}")

    return phonems_list


def get_transcripts_from_directory(directory_path=PATH_TRANSCRIPT_PHONEMIZED_GITHUB_DIR):
    aligned_transcripts = {}

    for filename in os.listdir(directory_path):
        if filename.endswith('phonemized_transcripts.txt'):
            with open(os.path.join(directory_path, filename), 'r') as f:
                content = f.read()

                file_id = os.path.splitext(filename)[0]

                aligned_transcripts[file_id] = content

    return aligned_transcripts


def get_duration_arrays_from_textgrids(directory_path=PATH_TRANSCRIPT_TEXTGRID):
    duration_arrays_dict = {}

    textgrid_files = [file for file in os.listdir(directory_path)
                      if file.endswith('.TextGrid')]

    for filename in textgrid_files:
        full_path = os.path.join(directory_path, filename)

        duration_array = extract_phonems_with_durations_from_textgrid(
            full_path)

        file_id = os.path.splitext(filename)[0]

        duration_arrays_dict[file_id] = duration_array

    return duration_arrays_dict


def get_phoneme_sequences_from_github_file(file_path=PATH_TRANSCRIPT_PHONEMIZED_GITHUB_FILE):
    phoneme_sequences_dict = {}

    with open(file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.split('|')
        sequence_id = parts[0]
        phoneme_sequence = parts[2].strip('{}').split(' ')
        phoneme_sequences_dict[sequence_id] = phoneme_sequence
    return phoneme_sequences_dict


def extract_phonems_with_durations_from_textgrid(file_path):
    '''
    return un dict qui array qui contient pour UNE SEULE SEQUENCE des tuples qui contiennent (phonem, duration)
    '''
    grid = textgrids.TextGrid(file_path)

    phoneme_tier = None
    for tier_name in grid:
        if tier_name in ["phones", "phonemes"]:
            phoneme_tier = grid[tier_name]
            break

    if phoneme_tier is None:
        raise ValueError(f"No phoneme tier found in {file_path}")

    phoneme_with_durations = []

    for interval in phoneme_tier:
        phoneme = interval.text.strip()
        if phoneme not in ["", "sp", "sil"]:
            duration = np.float32(interval.xmax - interval.xmin)
            phoneme_with_durations.append((phoneme, duration))

    return phoneme_with_durations


def get_phoneme_with_duration_dict(directory_path=PATH_TRANSCRIPT_TEXTGRID):
    '''
    return un dict avec key: sequence_id et value : array de tuples (phonem, duration)
    '''
    all_sequences_dict = {}

    textgrid_files = [file for file in os.listdir(directory_path)
                      if file.endswith('.TextGrid')]

    for filename in textgrid_files:
        full_path = os.path.join(directory_path, filename)

        phoneme_duration_list = extract_phonems_with_durations_from_textgrid(
            full_path)

        file_id = os.path.splitext(filename)[0]

        all_sequences_dict[file_id] = phoneme_duration_list

    return all_sequences_dict


def build_vocab(save_path=PATH_PHONES_MAPPING_NEW):
    unique_phonemes = set()
    phoneme_with_durations_dict = get_phoneme_with_duration_dict()
    for _, phoneme_duration_list in phoneme_with_durations_dict.items():
        phonemes = [tup[0] for tup in phoneme_duration_list]
        unique_phonemes.update(phonemes)

    # Ajoutez ici d'autres caractères ou symboles si nécessaire
    unique_phonemes.add('PAD')

    # Commencez l'indexation à partir de 1, car 0 est réservé pour le padding
    phoneme_to_index = {phoneme: index + 1 for index,
                        phoneme in enumerate(sorted(list(unique_phonemes)))}

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(phoneme_to_index, f, ensure_ascii=False, indent=4)


def load_phoneme_mapping(mapping_file_path=PATH_PHONES_MAPPING_NEW):
    with open(mapping_file_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    return mapping

# METHODE QUI CREE LES INPUT TOKENS POUR LE MODEL


def tokenize_pad_and_save_transcriptions(mapping_file_path=PATH_PHONES_MAPPING_NEW, save_path=PATH_PADDED_TOKENS):
    phoneme_to_token = load_phoneme_mapping(mapping_file_path)
    phoneme_with_duration_dict = get_phoneme_with_duration_dict()

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    max_len = max([len(seq) for seq in phoneme_with_duration_dict.values()])

    tokenized_and_padded_dict = {}

    for seq_id, phoneme_duration_list in phoneme_with_duration_dict.items():
        phonemes = [tup[0] for tup in phoneme_duration_list]
        token_ids = [phoneme_to_token[phoneme] for phoneme in phonemes]

        padded_token_ids = pad_sequences([token_ids],
                                         maxlen=max_len,
                                         padding='post',
                                         value=TOKEN_PADDING_VALUE)[0]

        tokenized_and_padded_dict[seq_id] = padded_token_ids

        np.save(os.path.join(
            save_path, f"{seq_id}_tokens.npy"), padded_token_ids)

    return tokenized_and_padded_dict

# METHODE QUI CREE LES INPUT PHONEM DURATIONS POUR LE MODEL


def pad_and_save_durations(save_path=PATH_PADDED_DURATIONS):
    phoneme_with_duration_dict = get_phoneme_with_duration_dict()
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    max_length = max([len(durations)
                     for durations in phoneme_with_duration_dict.values()])
    padded_durations_dict = {}

    for seq_id, phoneme_duration_list in phoneme_with_duration_dict.items():
        durations = [tup[1] for tup in phoneme_duration_list]
        padded_durations = pad_sequences([durations],
                                         maxlen=max_length,
                                         padding='post',
                                         value=DURATION_PADDING_VALUE,
                                         dtype='float32')[0]

        padded_durations_dict[seq_id] = padded_durations

        np.save(os.path.join(
            save_path, f"{seq_id}_phone_durations.npy"), padded_durations)

    return padded_durations_dict


def get_cleaned_transcriptions(transcriptions_dict):
    """
    Clean les transcriptions pour pouvoir les processer

    params : un dictionnaire de transcripts

    returns : un dictionnaire de clean_transcripts
    """
    def clean_transcription(transcription):
        punc_to_remove = string.punctuation
        lower = transcription.lower()
        without_punc = lower.translate(str.maketrans('', '', punc_to_remove))
        without_extra_spaces = " ".join(without_punc.split())
        return without_extra_spaces.strip()

    clean_transcriptions_dict = {}
    for sequence_id, transcription in transcriptions_dict.items():
        clean_transcriptions_dict[sequence_id] = clean_transcription(
            transcription)
    return clean_transcriptions_dict


def phonemize_transcripts(clean_trancripts_dict, separator=Separator(phone=' ', word='/')):
    """
    Convertit les transcriptions en liste de phonèmes à l'aide de la librairie phonemizer

    Parameters:
    - transcripts_dict : un dictionnaire avec pour clés les sequence_id et pour valeurs les transcriptions textuelles.

    Retour:
    - phonems_dict: dictionnaire avec les sequence_id comme clés et une liste de phonèmes en valeur.
    """
    transcriptions = clean_trancripts_dict.values()

    phonemized_transcriptions = phonemize(transcriptions,
                                          backend='espeak',
                                          language='en-us',
                                          separator=separator,
                                          strip=True)

    phonemized_lists = [transcription.split(separator.phone)
                        for transcription in phonemized_transcriptions]

    # split les phonèmes qui sont collés à cause de
    without_liaisons = []
    for phonemized_list in phonemized_lists:
        new_list = []
        for element in phonemized_list:
            split_elements = element.split(separator.word)
            new_list.extend(split_elements)
        without_liaisons.append(new_list)

    phonems_dict = {sequence_id: phonem_list
                    for sequence_id, phonem_list
                    in zip(clean_trancripts_dict.keys(), without_liaisons)}

    return phonems_dict


# def phonems_transcript_to_49(phonemized_transcriptions):
#     """
#     Gère les phonèmes multi-caractères dans les transcriptions pour n'avoir que 44 phonèmes

#     Paramètres:
#     - phonemized_transcriptions: Un dictionnaire avec sequence_id comme clés et une liste de phonèmes comme valeurs.

#     Retour:
#     - processed_transcripts: Un dictionnaire avec sequence_id comme clés et une liste traitée de phonèmes comme valeurs.
#     """

#     processed_transcripts = {}
#     for sequence_id, phonem_list in phonemized_transcriptions.items():
#         new_list = []
#         for phonem in phonem_list:
#             new_list.extend(handle_multi_char_phonem(phonem))
#         processed_transcripts[sequence_id] = new_list

#     return processed_transcripts


# def handle_multi_char_phonem(phonem):
#     '''
#     méthode qui prend en entrée un phonem et qui en ressort une liste de phonems
#     qui contiendra 1 ou plusieurs éléments.
#     Moche mais pragmatique.
#     '''
#     liste_44_phonems = ['aɪ', 'aʊ', 'b', 'd', 'eɪ', 'f', 'h', 'i', 'iə', 'j', 'k', 'l', 'm', 'n', 'o', 'oʊ', 'p', 's', 't', 'uː',
#                         'v', 'w', 'z', 'æ', 'ð', 'ŋ', 'ɑ', 'ɑː', 'ɔ', 'ɔɪ', 'ɔː', 'ə', 'ɚ', 'ɛ', 'ɡ', 'ɪ', 'ɹ', 'ɾ', 'ʃ', 'ʊ', 'ʊɹ', 'ʌ', 'ʒ', 'θ']

#     # si la longueur du phonème est de 1, tu retournes une liste qui contient uniquement le phonème
#     if len(phonem) == 1:
#         return [phonem]
#     # si la longueur du phonème est > à 2, tu retourne une liste de 2 éléments. Le premier élement contient les 2 premiers caractères, le 2ème élément contient le reste.
#     elif len(phonem) > 2:
#         return [phonem[:2], phonem[2:]]
#     # si la longueur du phonème est égale à 2:
#     elif len(phonem) == 2:
#         # s'il existe dans liste_44_phonems, tu renvoies un liste qui contient uniquement le phonèmes
#         if phonem in liste_44_phonems:
#             return [phonem]
#         # s'il est composé de 2 caractères qui existent individuellement dans liste_44_phonems alors tu le split en 2 et tu retournes une liste de 2 phonèmes
#         elif phonem[0] in liste_44_phonems and phonem[1] in liste_44_phonems:
#             return [phonem[0], phonem[1]]
#         # si le 1ere élement du phonème existe dans liste_44_phonems, renvoie une liste où il y a uniquement le premier phonème
#         elif phonem[0] in liste_44_phonems:
#             return [phonem[0]]
#     # sinon renvoie une liste vide
#     return []


# def tokenize_and_pad_transcripts(path_transcriptions_file=PATH_TRANSCRIPT_PHONEMIZED_GITHUB_FILE, path_mapping_phonem=PATH_PHONES_MAPPING_NEW):
#     transcripts_dict = get_phoneme_sequences_from_github_file(
#         path_transcriptions_file)
#     tokenized_transcriptions_dict = get_tokenized_transcripts(
#         transcripts_dict, path_mapping_phonem)

#     tokenized_transcriptions = list(tokenized_transcriptions_dict.values())

#     padded_lists = pad_sequences(
#         tokenized_transcriptions, padding='post', value=TOKEN_PADDING_VALUE)
#     padded_tokens_dict = {key: value for key, value in zip(
#         tokenized_transcriptions_dict.keys(), padded_lists)}

#     return padded_tokens_dict


# def get_input_sample(index):
#     files = sorted(os.listdir(PATH_PADDED_TOKENS))

#     if index < 0 or index >= len(files):
#         raise ValueError("Index out of range.")

#     file_path = os.path.join(PATH_PADDED_TOKENS, files[index])
#     key = files[index].strip("_tokens.npy")

#     seq_tokens = np.load(file_path, allow_pickle=True)

#     return {key: seq_tokens}


# def get_input_sample(index):
#     files_tokens = sorted(os.listdir(PATH_PADDED_TOKENS))
#     files_durations = sorted(os.listdir(PATH_PADDED_DURATIONS))

#     file_path_tokens = os.path.join(PATH_PADDED_TOKENS, files_tokens[index])
#     key = files_tokens[index].replace("_tokens.npy", "")

#     corresponding_duration_file = f"{key}_phone_durations.npy"
#     if corresponding_duration_file not in files_durations:
#         raise FileNotFoundError(
#             f"No file with this sequence_id in PATH_PADDED_DURATIONS")

#     file_path_durations = os.path.join(
#         PATH_PADDED_DURATIONS, corresponding_duration_file)

#     seq_tokens = np.load(file_path_tokens, allow_pickle=True)
#     seq_durations = np.load(file_path_durations, allow_pickle=True)

#     return {key: {'tokens': seq_tokens, 'durations': seq_durations}}
