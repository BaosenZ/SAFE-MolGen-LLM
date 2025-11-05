import os
import time
import re
import ast
import numpy as np
import pandas as pd
import joblib
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import rdMolDescriptors
from openai import OpenAI

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


def load_resources(eval_onlyGood):
    # Load all unique SMILES from original dataset in a list
    all_unique_smiles_path = (
        "../s0_prepData/s0f_analyzeData_LnAn/output_uniqueSMILES_ClassBySMARTS.xlsx"
    )
    all_exp_smiles_list = (
        pd.read_excel(all_unique_smiles_path)["SMILES"].to_numpy().flatten().tolist()
    )

    # Load descriptors of solvents
    conditions_path = "../s0_prepData/s0e_prepData_LnAn/db_LnAn_conditions_desp.xlsx"
    all_solvent_df = pd.read_excel(
        conditions_path, sheet_name="solvents_descriptors", index_col=0
    )
    all_solvent_df = all_solvent_df.iloc[:, :-2]

    # Load descriptors of metals
    all_metals_df = pd.read_excel(conditions_path, sheet_name="metals", index_col=0)
    all_metals_df = all_metals_df.iloc[:, :-1]

    # Load exp extractants evaluation
    if eval_onlyGood:
        original_smiles_eval_path = "./expSMILESeval_AmEu/expSMILESeval_onlyGood.xlsx"
    else:
        original_smiles_eval_path = "./expSMILESeval_AmEu/expSMILESeval.xlsx"
    original_smiles_eval_df = pd.read_excel(original_smiles_eval_path)

    # Load ML conditions features
    ml_conditions_columns_path = "resources/ML_features_conditions.xlsx"
    ml_conditions_df = pd.read_excel(ml_conditions_columns_path)
    ml_conditions_list = ml_conditions_df.iloc[:, 0].to_numpy().tolist()

    # Load all ML features
    all_ml_feature_columns_path = "resources/ML_features_all.xlsx"
    all_ml_feature_df = pd.read_excel(all_ml_feature_columns_path)
    all_ml_feature_series = all_ml_feature_df.iloc[:, 0]

    return (
        all_exp_smiles_list,
        all_solvent_df,
        all_metals_df,
        original_smiles_eval_df,
        ml_conditions_list,
        all_ml_feature_series,
    )


def callLLM_generate_smiles(
    smiles_eval_df,
    OPENAI_API_KEY,
    num_mol_per_message,
    metal1,
    metal2,
    tani_thr_RefExp_min,
    tani_thr_RefExp_max,
    tani_thr_RefGen_max,
    logP_threshold_min,
    design_focus,
    LLM_version_run,
):
    client = OpenAI(api_key=OPENAI_API_KEY)
    smiles_eval_json = smiles_eval_df.to_json(orient="records")
    messages = [
        {
            "role": "system",
            "content": f"""You are a coordination chemist with expertise in cheminformatics designing new extractant molecules for lanthanide and actinide separation; specifically for solvent extraction involving {metal1} (Target_metal) and {metal2} (Other_metal).
### **Response Format**  
- Return a **Python-formatted list** of SMILES strings such as ['SMILES_1', 'SMILES_2', 'SMILES_3']. Examples of valid SMILES strings are provided in the SMILES Evaluation Table.  
- Do not include explanations or additional text.""",
        },
        {
            "role": "user",
            "content": f"""Generate {num_mol_per_message} new SMILES strings for extractants which separate {metal1} from {metal2}. Generated molecules will be evaluated by the following criteria: 
1. **Distribution coefficient for the target metal {metal1}** should be high, corresponding to Target_metal = ORGANIC.
2. **Distribution coefficient for the other metal {metal2}** should be low, corresponding to Other_metal = AQUEOUS. 
3. **Similarity to SMILES where Source = Experimental**, as evaluated by Tanimoto similarity of ECFP fingerprint, should favor novel molecules with medium similarity in the range between {tani_thr_RefExp_min} and {tani_thr_RefExp_max}.
4. **Similarity to SMILES where Source = LLM generated**, as evaluated by Tanimoto similarity of ECFP fingerprint, should favor novel molecules with low or medium similarity smaller than {tani_thr_RefGen_max}.
5. **Organic/Water Partitioning (LogP)**, should favor LogP = ORGANIC phase with LogP value larger than {logP_threshold_min}. 

### **Goal**
- **Create novel extractant structures** where Target_metal = ORGANIC and Other_metal = AQUEOUS and LogP = ORGANIC by learning from the examples where **Source = Experimental** in the SMILES Evaluation Status Table.
- **Consider the following design focus:** {design_focus}.
- **Apply modifications** such as replacing, mixing, or changing side chains, functional groups, and/or backbone structures to ensure diversity.
- **Propose structurally diverse extractants** that vary from those in the table and meet all criteria and constraints above.
- **Propose easy to synthesize** and chemically accessible molecules.
- **Propose stable molecules** which are resistant to strong acids and radiolysis.

### **Current SMILES Evaluation Status Table (JSON Format):**
{smiles_eval_json}
""",
        },
    ]

    completion = client.chat.completions.create(
        model=LLM_version_run,
        messages=messages,
        temperature=1,
    )

    print(messages)
    print("Prompt tokens: ", completion.usage.prompt_tokens)
    print("Completion tokens: ", completion.usage.completion_tokens)

    message_new_smiles = completion.choices[0].message.content
    print("LLM message: ", message_new_smiles)
    python_block_pattern = r"```python\s*(.*?)\s*```"
    bracket_pattern = r"\[([^\[\]]*?(?:\[[^\[\]]*?\][^\[\]]*?)*?)\]"
    bracket_match = re.search(bracket_pattern, message_new_smiles, re.DOTALL)
    python_block_match = re.search(python_block_pattern, message_new_smiles, re.DOTALL)
    if python_block_match:
        smiles_list_str = python_block_match.group(1)
        try:
            message_new_smiles_list = ast.literal_eval(smiles_list_str)
        except (SyntaxError, ValueError) as e:
            print(f"Error parsing the list: {e}")
            message_new_smiles_list += [
                "LLM does not follow format"
            ] * num_mol_per_message
    elif bracket_match:
        smiles_list_str = bracket_match.group(0)
        try:
            message_new_smiles_list = ast.literal_eval(smiles_list_str)
        except (SyntaxError, ValueError) as e:
            print(f"Error parsing the list: {e}")
            message_new_smiles_list += [
                "LLM does not follow format"
            ] * num_mol_per_message
    else:
        print(f"LLM does not follow format.")
        message_new_smiles_list += ["LLM does not follow format"] * num_mol_per_message

    # Make sure the SMILES generated by LLM equals of number of molecules generated we specify
    if len(message_new_smiles_list) > num_mol_per_message:
        print("LLM does not generate number of SMILES we specify. ")
        message_new_smiles_list = message_new_smiles_list[0:num_mol_per_message]
    elif len(message_new_smiles_list) < num_mol_per_message:
        print("LLM does not generate number of SMILES we specify. ")
        num_smiles_add = num_mol_per_message - len(message_new_smiles_list)
        message_new_smiles_list += ["Invalid SMILES (Miss)"] * num_smiles_add

    return message_new_smiles_list


def generate_ml_input(
    oneSMILE, oneCondition_list, ml_conditions_list, all_ml_feature_series
):
    try:
        mol = Chem.MolFromSmiles(oneSMILE)
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=4, nBits=2048)
        rdkit_descriptors = [desc(mol) for _, desc in Descriptors.descList]
        ml_input = list(fingerprint) + rdkit_descriptors + oneCondition_list
    except Exception as e:
        print("Error: ", e)
        ml_input = [0] * (2048 + len(Descriptors.descList) + len(oneCondition_list))

    all_ml_series = pd.Series(
        ml_input,
        index=[f"CircularFP_{i}" for i in range(2048)]
        + [desc_name for desc_name, _ in Descriptors.descList]
        + ml_conditions_list,
    )

    ml_input = pd.Series(all_ml_series, index=all_ml_feature_series).to_numpy().tolist()

    return ml_input


def similarStructureCheck(one_smile, all_smiles_list):
    valid_all_smiles = []
    for one_smiles in all_smiles_list:
        m = rdkit.Chem.MolFromSmiles(one_smiles)
        if m:
            valid_all_smiles.append(one_smiles)

    similarity_score_list = []
    max_similarity_score = 0

    try:
        m1 = rdkit.Chem.MolFromSmiles(one_smile)
        fp1 = rdkit.Chem.AllChem.GetMorganFingerprintAsBitVect(m1, 4, nBits=2048)
        for one_smile_ori in valid_all_smiles:
            m2 = rdkit.Chem.MolFromSmiles(one_smile_ori)
            fp2 = rdkit.Chem.AllChem.GetMorganFingerprintAsBitVect(m2, 4, nBits=2048)
            similarity = rdkit.DataStructs.TanimotoSimilarity(fp1, fp2)
            similarity_score_list.append(similarity)
        max_similarity_score = max(similarity_score_list)
    except Exception as e:
        print("Error: ", e)
        max_similarity_score = 0
        similarity_score_list = [0] * len(all_smiles_list)

    return max_similarity_score, similarity_score_list


def classify_D(ML_output):
    if ML_output == 2:
        return "ORGANIC"
    elif ML_output == 1:
        return "UNSELECTIVE"
    else:
        return "AQUEOUS"


def extract_metal_label(OUA_tuple_list):
    # Define custom sort order based on (Target_metal, Other_metal) combinations
    combo_order = {
        ("ORGANIC", "AQUEOUS"): 0,
        ("ORGANIC", "UNSELECTIVE"): 1,
        ("UNSELECTIVE", "AQUEOUS"): 2,
        ("ORGANIC", "ORGANIC"): 3,
        ("UNSELECTIVE", "UNSELECTIVE"): 4,
        ("AQUEOUS", "AQUEOUS"): 5,
        ("UNSELECTIVE", "ORGANIC"): 6,
        ("AQUEOUS", "UNSELECTIVE"): 7,
        ("AQUEOUS", "ORGANIC"): 8,
    }

    # Score oua tuple list
    score_dic_list = []
    for combo in OUA_tuple_list:
        score = combo_order.get(combo)
        score_dic_list.append({combo: score})

    # Extract metal label from score
    flattened = []
    for d in score_dic_list:
        for key, value in d.items():
            flattened.append((key, value))
    min_entry = min(flattened, key=lambda x: x[1])
    (metal1_label, metal2_label), score = min_entry

    return metal1_label, metal2_label
