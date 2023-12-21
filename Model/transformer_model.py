from transformers import AutoTokenizer, AutoModel
from preprocessing import load_train_data, load_eval_data
import sklearn

miniLM_tokenizer = AutoTokenizer.from_pretrained("microsoft/Multilingual-MiniLM-L12-H384")
miniLM_model = AutoModel.from_pretrained("microsoft/Multilingual-MiniLM-L12-H384")


def get_batches(batch_size, text_pairs, labels=None, shuffle=True):
    total_data_size = len(text_pairs)
    index_ls = [i for i in range(total_data_size)]

    if shuffle:
        dataset = sklearn.utils.shuffle(index_ls)

    if labels:
        for start_i in range(0, total_data_size, batch_size):
            # get batch_texts, batch_labels
            end_i = min(total_data_size, start_i + batch_size)
            batch_text_pairs = text_pairs[start_i:end_i]
            batch_labels = labels[start_i:end_i]
            yield batch_text_pairs, batch_labels

    else:
        for start_i in range(0, total_data_size, batch_size):
            # get batch_texts
            end_i = min(total_data_size, start_i + batch_size)
            batch_text_pairs = text_pairs[start_i:end_i]
            yield batch_text_pairs


if __name__ == "__main__":
    train_file_eng = '../data/TrackA_data/eng/eng_train.csv'
    eval_file_afr = '../data/TrackC_data/afr/afr_pilot.csv'
    train_pairs_eng, train_scores_eng = load_train_data(train_file_eng)
    eval_pairs_afr = load_eval_data(eval_file_afr)

    train_batches_eng = list(get_batches(5, train_pairs_eng, shuffle=True))
    encoded_input = tokenizer(train_batches_eng[0], padding=True, return_tensors="pt")
    print(encoded_input)
