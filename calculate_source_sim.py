import sys
import pandas as pd
from itertools import combinations
from tqdm import tqdm
import rbo 

def calculate_jaccard(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union
    
def calculate_rbo(lst1, lst2, p):
    # Note that I'm not setting k (=depth of evaluation) here, which means that it is set to indefinite (i.e. entirety of both lists)
    result = rbo.RankingSimilarity(lst1,lst2).rbo(p=p) 
    return result

def calculate_pairwise_ji_rbo(df, target_col, p_values):
    grouped = df.groupby('run_uid')[target_col].apply(list).reset_index()
    
    sim = []
    for run_uid1, run_uid2 in tqdm(combinations(grouped['run_uid'], 2)):
        # Extract the sources for the two run_uids
        sources1 = grouped[grouped['run_uid'] == run_uid1][target_col].iloc[0]
        sources2 = grouped[grouped['run_uid'] == run_uid2][target_col].iloc[0]
        
        # remove NaN values.
        sources1 = list(set([u for u in sources1 if pd.notna(u)]))
        sources2 = list(set([u for u in sources2 if pd.notna(u)]))
        
        # Calculate RBO 
        values={}
        for p in p_values:
            rbo = calculate_rbo(sources1, sources2, p=p)
            column_name = "rbo" + str(p) + "_value"
            values.update({column_name:rbo})
    
        ji_value = calculate_jaccard(sources1, sources2)
        values.update({"ji_value":ji_value})
    
        # conditions
        user_input1 = df[df['run_uid']==run_uid1]['user_input'].iloc[0]
        user_input2 = df[df['run_uid']==run_uid2]['user_input'].iloc[0]
        search_history1 = df[df['run_uid']==run_uid1]['search_history'].iloc[0]
        search_history2 = df[df['run_uid']==run_uid2]['search_history'].iloc[0]
    
        results = {'run_uid1': run_uid1, 'run_uid2': run_uid2, 
                   'user_input1': user_input1, 'user_input2': user_input2, 'search_history1': search_history1, 'search_history2': search_history2}
        results.update(values)
    
        # Store the result
        sim.append(results)
    
    sim_df = pd.DataFrame(sim)
    
    return sim_df

if __name__ == "__main__":
    
    # define input and output files
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # read input file
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)

    # Define the target column and p_values
    target_column = 'domain' 
    p_values = [0.8, 0.95]  
    
    # Calculate similarities
    similarities = calculate_pairwise_ji_rbo(df, target_column, p_values)
    
    # write dataframe
    similarities.to_csv(output_file, index=False)
