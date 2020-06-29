from torch.utils.data import TensorDataset
import logging
import torch
import copy
import json
import os

logger = logging.getLogger(__name__)

SEP = '[SEP]'


def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]
    return all_input_ids, all_attention_mask, all_token_type_ids, all_labels


def load_and_cache_examples(config, model_name, tokenizer, data_type='train'):
    dataset = Dataset()
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        config.data_dir, 'cached_{}_{}_{}'.format(
            data_type,
            model_name,
            str(config.max_seq_length)
        )
    )
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", config.data_dir)
        label_list = dataset.get_labels()

        if data_type == 'train':
            examples = dataset.get_train_examples(config.train_path)
        elif data_type == 'dev':
            examples = dataset.get_dev_examples(config.dev_path)
        else:
            examples = dataset.get_test_examples(config.test_path)

        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=config.max_seq_length,
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=0
        )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long, device=config.device)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long, device=config.device)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long, device=config.device)
    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long, device=config.device)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long, device=config.device)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels)
    return dataset


def convert_examples_to_features(
        examples,
        tokenizer,
        max_length=512,
        label_list=None,
        pad_token=0,
        pad_token_segment_id=0,
        mask_padding_with_zero=True):
    """
    Loads a data file into a list of ``InputFeatures``
    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % ex_index)

        inputs = tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        input_len = len(input_ids)
        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)

        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length"
        assert len(attention_mask) == max_length, "Error with input length"
        assert len(token_type_ids) == max_length, "Error with input length"

        label = label_map[example.label]
        features.append(InputFeatures(input_ids, attention_mask, token_type_ids, label, input_len))
    return features


class Dataset(object):

    """ Dataset for the TNEWS """

    @classmethod
    def _read_json(cls, input_file):
        """Reads a json list file."""
        with open(input_file, "r") as f:
            reader = f.readlines()
            return [json.loads(line.strip()) for line in reader]

    @classmethod
    def _create_examples(cls, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line['sentence'] + SEP + line['keywords'] if line['keywords'].strip() else line['sentence']
            text_b = None
            label = str(line['label'])
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    @classmethod
    def get_labels(cls):
        return [str(100 + i) for i in range(17) if i != 5 and i != 11]

    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_json(data_dir), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_json(data_dir), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_json(data_dir), "test")


class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, attention_mask, token_type_ids, label, input_len):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.input_len = input_len
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
