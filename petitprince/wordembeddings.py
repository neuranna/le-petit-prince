import numpy as np
import pandas as pd


def get_embedding_dict(
        emb_txt: str = '../stimulus-info/lpp_en_emb.txt'
) -> dict:
    """
    load dictionary that maps words to embeddings
    each key is a word, each value is a 1d array of shape (300,)
    """
    word_embs_dict = dict()
    with open(emb_txt, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.replace('\n', '')
        linelist = line.split(' ')
        embedding = np.array(linelist[1:], dtype=np.single)
        word_embs_dict[linelist[0]] = embedding
    return word_embs_dict


def get_unique_words(transcript_csv: str = '../stimulus-info/lpp_en_snt_nopunct.csv') -> set:
    """Mostly for sanity checking"""
    sentences = pd.read_csv(transcript_csv, header=None).to_numpy()
    words = ' '.join([s for s in sentences.flatten()]).split(' ')
    unique_words = set(words)
    return unique_words


def generate_wordembs_trs(
        wordonsets_csv = '../stimulus-info/lpp_en_regressors.csv',
        emb_txt ='../stimulus-info/lpp_en_emb.txt',
        save_to_file=True,
        scans_per_run=282,
        tr = 2.,
):
    word_embs_dict = get_embedding_dict(emb_txt)
    wordonsets = pd.read_csv(wordonsets_csv)
    ons_diffs = np.pad(wordonsets.onset.to_numpy(), (1, 0)) - np.pad(wordonsets.onset.to_numpy(), (0, 1))
    runbreaks = np.where(ons_diffs > 0)[0]
    start = 0
    runwise_onsets = []
    for runbreak in runbreaks:
        onsets_thisrun = wordonsets[start:runbreak]
        start = runbreak
        runwise_onsets.append(onsets_thisrun)
    runwise_words, runwise_embeddings = [], []
    for run_i, onsets in enumerate(runwise_onsets):
        embeddings, words_ = [], []
        for tr_i in range(scans_per_run):
            tr_on, tr_off = tr_i * tr, tr_i * tr + tr
            words = onsets[(onsets['onset'] >= tr_on) & (onsets['onset'] < tr_off)]['word']
            embs = np.zeros(300)
            for word in words:
                try:
                    embs += word_embs_dict[word]
                except KeyError:
                    continue
            embeddings.append(embs)
            words_.append(words.to_list())
        runwise_embeddings.append(embeddings)
        runwise_words.append(words_)
    runwise_embeddings = [np.array(embeddings) for embeddings in runwise_embeddings]
    if save_to_file:
        for run_i, (words, emb) in enumerate(zip(runwise_words, runwise_embeddings)):
            np.savetxt(f'embeddings_en_TRbinned_run-{run_i+1}.txt', emb)
            pd.DataFrame(words).to_csv(f'words_en_TRbinned_run-{run_i + 1}.csv', sep=',')
    return runwise_embeddings, runwise_words

# if __name__=='__main__':
    # _,_ = generate_wordembs_trs()