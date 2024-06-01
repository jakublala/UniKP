from __future__ import annotations
import torch
from build_vocab import WordVocab
from pretrain_trfm import TrfmSeq2seq
from utils import split
from transformers import T5EncoderModel, T5Tokenizer
import re
import gc
import numpy as np
import pandas as pd
import pickle
import math

device = torch.device('mps')

def smiles_to_vec(
    Smiles: list[str], vocab_path: str, trfm_model_path: str
) -> np.ndarray:
    """
    Converts SMILES strings to vectors using a transformer model.
    Args:
    - Smiles (list[str]): A list of SMILES (Simplified Molecular Input Line Entry System) strings.
    - vocab_path (str): Path to the vocabulary pickle file.
    - trfm_model_path (str): Path to the transformer model file.
    Returns:
    - np.ndarray: A numpy array of vectors representing the SMILES strings.
    """
    pad_index, unk_index, eos_index, sos_index = 0, 1, 2, 3
    vocab = WordVocab.load_vocab(vocab_path)

    def get_inputs(sm):
        seq_len = 220
        sm = sm.split()
        sm = sm[:109] + sm[-109:] if len(sm) > 218 else sm
        ids = [vocab.stoi.get(token, unk_index) for token in sm]
        ids = [sos_index] + ids + [eos_index]
        padding = [pad_index] * (seq_len - len(ids))
        ids.extend(padding)
        seg = [1] * len(ids)
        seg.extend(padding)
        return ids, seg

    def get_array(smiles):
        x_id, x_seg = [], []
        for sm in smiles:
            a, b = get_inputs(sm)
            x_id.append(a)
            x_seg.append(b)
        return torch.tensor(x_id), torch.tensor(x_seg)

    trfm = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
    trfm.load_state_dict(torch.load(trfm_model_path, map_location=device))
    trfm.eval()

    x_split = [split(sm) for sm in Smiles]
    xid, xseg = get_array(x_split)
    X = trfm.encode(torch.t(xid))
    
    assert X.shape[1] == 1024, f"Expected 1024 features, got {X.shape[1]} features."

    return X


def Seq_to_vec(sequences: list[str], model_path: str) -> np.ndarray:
    """
    Converts protein sequences to vectors using a T5 Encoder model.
    Args:
    - sequences (list[str]): A list of protein sequences.
    - model_path (str): Path to the directory containing the T5 model and tokenizer.
    Returns:
    - np.ndarray: A numpy array of normalized feature vectors representing the sequences.
    """

    # Truncate sequences longer than 1000 characters
    sequences = [
        seq[:500] + seq[-500:] if len(seq) > 1000 else seq for seq in sequences
    ]

    # Add spaces between each character for tokenization
    sequences = [" ".join(seq) for seq in sequences]

    # Replace unknown amino acids with 'X'
    sequences = [re.sub(r"[UZOB]", "X", seq) for seq in sequences]

    # Load tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_path)
    model.to(device).eval()
    gc.collect()

    # Process sequences and get embeddings
    features = []
    for seq in sequences:
        ids = tokenizer.batch_encode_plus([seq], add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids["input_ids"]).to(device)
        attention_mask = torch.tensor(ids["attention_mask"]).to(device)
        with torch.no_grad():
            embedding = (
                model(input_ids=input_ids, attention_mask=attention_mask)
                .last_hidden_state.cpu()
                .numpy()
            )
        seq_len = (attention_mask == 1).sum()
        seq_emd = embedding[0][: seq_len - 1]  # Get embeddings for each sequence
        features.append(seq_emd)

    # Normalize features
    features_normalize = np.array([emd.mean(axis=0) for emd in features])

    assert features_normalize.shape[1] == 1024, f"Expected 1024 features, got {features_normalize.shape[1]} features."

    return features_normalize


def get_protein_ligand_representation(
    sequences: list[str],
    Smiles: list[str],
    seq_model_path: str,
    smiles_vocab_path: str,
    smiles_model_path: str,
) -> np.ndarray:
    """
    Converts protein sequences and SMILES strings to their vector representations and concatenates them.
    Args:
    - sequences (list[str]): A list of protein sequences.
    - Smiles (list[str]): A list of SMILES (Simplified Molecular Input Line Entry System) strings.
    - seq_model_path (str): Path to the directory containing the T5 model for sequences.
    - smiles_vocab_path (str): Path to the vocabulary file for SMILES.
    - smiles_model_path (str): Path to the transformer model file for SMILES.
    Returns:
    - np.ndarray: A concatenated numpy array of the vector representations of sequences and SMILES.
    """
    # Get vector representations for sequences and SMILES
    seq_vec = Seq_to_vec(sequences, seq_model_path)
    smiles_vec = smiles_to_vec(Smiles, smiles_vocab_path, smiles_model_path)

    # Concatenate the vectors
    fused_vector = np.concatenate((smiles_vec, seq_vec), axis=1)
    return fused_vector


def predict_catalytic_properties(
    sequences: list[str],
    Smiles: list[str],
    fused_vector: np.ndarray,
    model_paths: dict[str:str] = None,
) -> pd.DataFrame:
    """
    Predicts the catalytic properties (kcat, Km, and kcat/Km) for given sequences and SMILES.
    Args:
    - sequences (List[str]): A list of protein sequences.
    - Smiles (List[str]): A list of SMILES strings.
    - fused_vector (np.ndarray): The concatenated numpy array of vector representations of sequences and SMILES.
    - model_paths (dict[str:str]): A dictionary of paths to the models for each catalytic property.
    Note: Original model outputs log10 values of the catalytic properties.
          These are transformed here to their original space.
    Returns:
    - pd.DataFrame: A dataframe containing the sequences, SMILES, and their predicted catalytic properties.
    """
    properties = ["kcat", "KM", "kcat/KM"]
    if model_paths is None:
        model_paths = {
            "kcat": "UniKP/UniKP_for_kcat.pkl",
            "KM": "UniKP/UniKP_for_Km.pkl",
            "kcat/KM": "UniKP/UniKP_for_kcat_Km.pkl",
        }
    results = []
    for prop in properties:
        with open(model_paths[prop], "rb") as f:
            model = pickle.load(f)
        Pre_label = model.predict(fused_vector)
        Pre_label_pow = [math.pow(10, Pre_label[i]) for i in range(len(Pre_label))]
        results.append(Pre_label_pow)

    # Creating DataFrame
    res = pd.DataFrame(
        {
            "sequences": sequences,
            "Smiles": Smiles,
            "kcat [1/s]": results[0],
            "KM [mM]": results[1],
            "kcat/KM [1/mM.s]": results[2],
        }
    )
    return res


if __name__ == "__main__":
    # Example usage
    sequences = [
        "MEDIPDTSRPPLKYVKGIPLIKYFAEALESLQDFQAQPDDLLISTYPKSGTTWVSEILDMIYQDGDVEKCRRAPVFIRVPFLEFKA"
        "PGIPTGLEVLKDTPAPRLIKTHLPLALLPQTLLDQKVKVVYVARNAKDVAVSYYHFYRMAKVHPDPDTWDSFLEKFMAGEVSYGSW"
        "YQHVQEWWELSHTHPVLYLFYEDMKENPKREIQKILKFVGRSLPEETVDLIVQHTSFKEMKNNSMANYTTLSPDIMDHSISAFMRK"
        "GISGDWKTTFTVAQNERFDADYAKKMEGCGLSFRTQL"
    ]
    Smiles = ["OC1=CC=C(C[C@@H](C(O)=O)N)C=C1"]
    fused_vector = get_protein_ligand_representation(
        sequences,
        Smiles,
        "prot_t5_xl_uniref50",
        "vocab.pkl",
        "trfm_12_23000.pkl",
    )
    res = predict_catalytic_properties(sequences, Smiles, fused_vector)
    res.to_csv("Kinetic_parameters_predicted_label.csv")
