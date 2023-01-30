import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import xgboost as xgb


def normalize(text, stemmer=None, reverse_dict=None):
    '''normalize text to be used in machine learning applications
    
    The text will be processed sentence by sentence and will be
    tokenized, removing punctuation, numbers and so on.  The resulting
    words will be stemmed if a stemmer is provided.
    
    Parameters
    ----------
    text: str
        Multiline string representation of the text to process
    stemmer: nltk.stem.x.y
        NLTK stemmer to use
    reverse_dict: dict
        Dictionary to add the words to that are stemmed.  The
        key is the stem, the corresponding value a set of words
        that have the key as stem
    
    Returns
    -------
    words: list
        A list of words
    '''
    if stemmer is None:
        stemmer = nltk.stem.snowball.EnglishStemmer()
    words = []
    for sentence in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sentence):
            if word.isalpha():
                stem = stemmer.stem(word.lower())
                words.append(stem)
                if reverse_dict is not None:
                    if stem not in reverse_dict:
                        reverse_dict[stem] = {word}
                    else:
                        reverse_dict[stem].add(word)
    return words

def create_output(data, col_name, item_id):
    output_col_name = 'output'
    output = data[['EFSAID', col_name]].drop_duplicates().query(f'{col_name} == {item_id}')
    output.rename(columns={col_name: output_col_name}, inplace=True)
    output.output = 1
    neg_ids = data[~data.EFSAID.isin(output.EFSAID)].EFSAID.unique()
    neg_output = pd.DataFrame({
        'EFSAID': neg_ids,
        output_col_name: [0]*len(neg_ids),
    })
    return pd.concat([output, neg_output], axis=0) \
             .sort_values(by=['EFSAID']) \
             .reset_index(drop=True)

def split_indices(output, fraction):
    pos_output_1 = output[output.output == 1].sample(frac=fraction)
    neg_output_1 = output[output.output == 0].sample(frac=fraction)
    output_1 = pd.concat([pos_output_1, neg_output_1], axis=0).sort_values(by=['EFSAID'])
    output_2 = output[~output.EFSAID.isin(output_1.EFSAID)].sort_values(by=['EFSAID'])
    return output_1.index.values, output_2.index.values

def create_dataset(tf_idf, output_df, indices):
    input = tf_idf[indices, :]
    output = output_df.iloc[indices].output.to_numpy()
    return input, output

def create_datasets(input_data, output_data, value_name, value,
                    test_frac=0.2, val_frac=0.2, verbose=False):
    value_output = create_output(output_data, value_name, value)
    test_indices, training_indices = split_indices(value_output, test_frac)
    val_indices, train_indices = split_indices(value_output.iloc[training_indices],
                                               val_frac)
    test_input, test_output = create_dataset(input_data, value_output, test_indices)
    np.savetxt(f'data/6-validations/{value_name}_{value}_target_output.txt',
               test_output)
    val_input, val_output = create_dataset(input_data, value_output, val_indices)
    train_input, train_output = create_dataset(input_data, value_output,
                                               train_indices)
    with open('data/6-validations/indices.txt', 'w') as file:
        for index in train_indices:
            print(f'{index},training', file=file)
        for index in val_indices:
            print(f'{index},validation', file=file)
        for index in test_indices:
            print(f'{index},test', file=file)
    if verbose:
        print(f'training data:   {train_input.shape} -> {train_output.shape}', file=sys.stderr)
        print(f'validation data: {val_input.shape} -> {val_output.shape}', file=sys.stderr)
        print(f'test data:       {test_input.shape} -> {test_output.shape}', file=sys.stderr)
    return (
        xgb.DMatrix(train_input, label=train_output),
        xgb.DMatrix(val_input, label=val_output),
        xgb.DMatrix(test_input)
    )

def save_progress(progress, file_name):
    train_progress = progress['train']['auc']
    eval_progress = progress['eval']['auc']
    time_progress = range(1, len(train_progress) + 1)
    progress_data = np.array([time_progress, train_progress, eval_progress]).transpose()
    np.savetxt(file_name, progress_data)

def train_eval(input_data, output_data, value_name, value,
               param, nr_steps, verbose=False):
    train_data, val_data, test_data = create_datasets(input_data, output_data,
                                                      value_name, value,
                                                      verbose=verbose)
    evallist = [(train_data, 'train'), (val_data, 'eval')]
    progress = {}
    bst = xgb.train(param, train_data, nr_steps, evals=evallist,
                    evals_result=progress, verbose_eval=False)
    bst.save_model(f'models/{value_name}_{value}_model.json')
    save_progress(progress, f'data/6-validations/{value_name}_{value}_auc.txt')
    test_output = bst.predict(test_data)
    np.savetxt(f'data/6-validations/{value_name}_{value}_output.txt',
               test_output)
    all_data = xgb.DMatrix(input_data)
    all_output = bst.predict(all_data)
    np.savetxt(f'data/5-predictions/{value_name}_{value}_output.txt',
               all_output)

def evaluate(value_name, value):
    output = np.genfromtxt(f'data/6-validations/{value_name}_{value}_output.txt')
    target_output = np.genfromtxt(f'data/6-validations/{value_name}_{value}_target_output.txt')
    return np.count_nonzero(target_output - output.round(0))/len(output)

def confustion_matrix(value_name, value):
    output = np.genfromtxt(f'data/6-validations/{value_name}_{value}_output.txt')
    target_output = np.genfromtxt(f'data/6-validations/{value_name}_{value}_target_output.txt')
    df = pd.DataFrame({
        'target': target_output,
        'output': output,
    })
    true_pos_out = len(df[(df.target > 0.5) & (df.output > 0.5)])
    false_pos_out = len(df[(df.target <= 0.5) & (df.output > 0.5)])
    true_neg_out = len(df[(df.target <= 0.5) & (df.output <= 0.5)])
    false_neg_out = len(df[(df.target > 0.5) & (df.output <= 0.5)])
    return [
        [true_pos_out, false_pos_out],
        [false_neg_out, true_neg_out],
    ]

def plot_score(file_name):
    data = np.genfromtxt(file_name, delimiter=',', names=True)
    name = data.dtype.names[0]
    plt.bar(data[name], data['score'])