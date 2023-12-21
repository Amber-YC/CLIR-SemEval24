#自建adapter
from adapters import AutoAdapterModel

model = AutoAdapterModel.from_pretrained("roberta-base")
model.add_adapter("adapter1", config="seq_bn") # add the new adapter to the model
model.add_classification_head("adapter1", num_classes=3) # add a sequence classification head

model.train_adapter("adapter1") # freeze the model weights and activate the adapter

# 引用adapter。la_adapters trained by https://adapterhub.ml/adapters/AdapterHub/xmod-base-af_ZA/
from adapters import AutoAdapterModel

model = AutoAdapterModel.from_pretrained("AdapterHub/xmod-base")
la_adapter = model.load_adapter("AdapterHub/xmod-base-af_ZA", source="hf", set_active=True)

model.set_active_adapters(la_adapter)


