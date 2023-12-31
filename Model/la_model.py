from transformers import AutoTokenizer, AutoModel
from preprocessing import load_data
from sklearn.model_selection import train_test_split
import sklearn
from adapters import AutoAdapterModel, AdapterConfig, AdapterType


# Pre-trained transformer
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
model = AutoAdapterModel.from_pretrained("bert-base-multilingual-cased")

# Add la layer
config = AdapterConfig.load("pfeiffer", non_linearity="relu", reduction_factor=2)
la_model = model.load_adapter("en/wiki@ukp", config=config)
model.set_active_adapters(la_model)



if __name__ == "__main__":
    train_file_eng = '../Semantic_Relatedness_SemEval2024-main/Track A/eng/eng_train.csv'
    #eval_file_afr = '../data/TrackC_data/afr/afr_pilot.csv'
    train_pairs_eng, train_scores_eng = load_train_data(train_file_eng)
    #eval_pairs_afr = load_eval_data(eval_file_afr)

    texts_train, texts_val, labels_train, labels_val = train_test_split(train_pairs_eng, train_scores_eng,
                                                                        test_size=0.2, random_state=42)
    train_batches_eng = list(get_batches(16, texts_train, labels=labels_train, shuffle=True))
    #print(train_batches_eng[0])

    # usage of Transformer
    #encoded_input = miniLM_tokenizer(train_batches_eng[0][0], padding=True, return_tensors="pt")
    encoded_input = tokenizer(train_batches_eng[0][0], padding=True, return_tensors="pt") # train_batches_eng[0][0]: 第0个batch的batch_text_pairs

    output = model(**encoded_input)

    print(output)  # last_hidden_state.shape = ([16(batch_size),82(padded sequence length),384(hidden_size)])
