import sys
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from itertools import combinations
from tqdm import tqdm

def create_embeddings(text, share, model):
    em = model.encode(" ".join(text[:int(len(text) * share)]))
    return em 

def calculate_cosine_similarity(run_uid1, run_uid2, em1, em2, df):
    
    # calculate cosine similarities
    cos_sim = util.cos_sim(em1, em2).item()
    
    # conditions
    user_input1 = df[df['run_uid']==run_uid1]['user_input'].iloc[0]
    user_input2 = df[df['run_uid']==run_uid2]['user_input'].iloc[0]
    search_history1 = df[df['run_uid']==run_uid1]['search_history'].iloc[0]
    search_history2 = df[df['run_uid']==run_uid2]['search_history'].iloc[0]
    
    
    results = {'run_uid1': run_uid1, 'run_uid2': run_uid2, 
               'user_input1': user_input1, 'user_input2': user_input2, 'search_history1': search_history1, 'search_history2': search_history2,
               'cos_value':cos_sim}
    
    return results


if __name__ == "__main__":
    
    # define input and output files
    input_file = sys.argv[1]
    output_file = sys.argv[3]
    
    # read input file
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)

    # Define the target column and p_values
    target_col = 'text_for_sim'
    share = float(sys.argv[2]) # truncate SERP to e.g., 0.33 - set to 1 if whole list should be considered
    
    # choice of transformer model
    modelchoice = 'sentence-transformers/distiluse-base-multilingual-cased' # more general
    #modelchoice = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1' # for Q&A
    model = SentenceTransformer(modelchoice)
   
    # create list of texts per run_uid
    #grouped = df.groupby('run_uid')[target_col].apply(list).reset_index()
    
    # create embeddings for all run_uids only once
    print('Start calculating embeddings')
    embeddings = {}
    for run_uid in tqdm(df['run_uid'].unique()):
        #text = grouped[grouped['run_uid'] == run_uid][target_col].iloc[0]
        text = df.loc[df['run_uid'] == run_uid, target_col].tolist()
        # remove nan values
        text = [t for t in text if pd.notna(t)]
        
        em = create_embeddings(text, share, model)
        embeddings[run_uid] = em
    
    # calculate cosine similarities
    print('Start calculating cosine similarities')
    sim = []
    for run_uid1, run_uid2 in tqdm(combinations(df['run_uid'].unique(), 2)):
        em1 = embeddings[run_uid1]
        em2 = embeddings[run_uid2]
        results = calculate_cosine_similarity(run_uid1, run_uid2, em1, em2, df)
        sim.append(results)

    similarities = pd.DataFrame(sim)
    
    # write dataframe
    similarities.to_csv(output_file, index=False)

    
    
