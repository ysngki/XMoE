# Outline

To reproduce XMoE, follow these steps:

1. Clone this repository.
2. Set up the environment.
3. Download the datasets.
4. Preprocess the datasets.
5. Execute the provided scripts.

# Environment

As per the instructions from Megatron-LM, the following dependencies are required: `apex`, `NCCL`. Additionally, `deepseep`, `torch` are also necessary.

# Datasets

## Download

Below are the links to download the pretraining datasets:

- [Openwebtext](https://skylion007.github.io/OpenWebTextCorpus/)
- [Bookcorpus](https://twitter.com/theshawwn/status/1301852133319294976)
- [Wikitext 103](https://huggingface.co/datasets/wikitext?row=2) from Huggingface
- **English Wikipedia** and **CC News** datasets are also available on Hugging Face.

## Preprocess

> The training data requires preprocessing. First, place your training data in a loose json format, with one json containing a text sample per line. For example:
>
> ```json
> {"src": "www.nvidia.com", "text": "The quick brown fox", "type": "Eng", "id": "0", "title": "First Part"}
> {"src": "The Internet", "text": "jumps over the lazy dog", "type": "Eng", "id": "42", "title": "Second Part"}
> ```
>
> The name of the `text` field of the json can be changed by using the `--json-key` flag in [`preprocess_data.py`](./tools/preprocess_data.py) The other metadata are optional and are not used in training.
>
> The loose json is then processed into a binary format for training. To convert the json into mmap format use `preprocess_data.py`.

Here is an example script for data preparation:

```bash
python tools/preprocess_data.py \
	--input ../pretrain_data/cc_news.json \
	--output-prefix gpt2-cc_news \
	--vocab-file dataset/gpt2-vocab.json \
	--tokenizer-type GPT2BPETokenizer \
	--merge-file dataset/gpt2-merges.txt \
	--append-eod --workers 8
```

The `vocab-file` and `merge-file` can be obtained by running `dataset/download_vocab.sh`.

The script will generate two files: `gpt2-cc_news_text_sentence.bin` and `gpt2-cc_news_text_sentence.idx`.

To merge multiple datasets into one, use the following command:

```bash
python tools/merge_datasets.py --input temp_datasets_are_here/ --output-prefix merged-dataset
```

# Training

To start training, use the provided script example:

 ```bash
 cd MoE-Megatron-DeepSpeed/examples_deepspeed/MoE/
 bash last_run.sh
 ```

Below are some important arguments to note:

```bash
MOE_FFN_HIDDEN_SIZE=384 # conrtol the expert size
EP_SIZE=64 # number of experts
EP_INTERVAL=2
TOPK=8
THRESHOLD=0.9
MOE_TRAIN_CAP_FACTOR=8.0
MOE_EVAL_CAP_FACTOR=8.0

VOCAB_PATH=MoE-Megatron-DeepSpeed/dataset/gpt2-vocab.json
MERGE_PATH=MoE-Megatron-DeepSpeed/dataset/gpt2-merges.txt
DATA_PATH=yuanhang/pretrain_data/gpt2-dataset_text_document
```

# Details

The XMoE implementation can be found primarily in s```MoE-Megatron-DeepSpeed/megatron/model/mlp_gating.py``` and ```MoE-Megatron-DeepSpeed/megatron/model/transformer.py```.







