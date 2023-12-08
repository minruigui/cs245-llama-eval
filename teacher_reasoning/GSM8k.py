import openai
from data.completion_dataset import CompletionMetadata, CompletionDataset
from oai.inference import infer_completion_data
from data.completion_dataset import CompletionIdentifier
from data.split import load_train_test_split
from evaluation.evaluator import Evaluator
from evaluation.summary import summarize_evaluation
from oai.finetune import init_finetune, generate_finetune_data_from_completion_dataset
from oai.utils.api_wrapper import fetch_model_ids
import json
from paths import get_finetune_data_path
from oai.utils.metadata import get_file_id, get_finetune_id, get_model_id, get_model_key


openai.api_key = ""
teacher_base_model = "text-davinci-002"
base_model = "babbage"
dataset_key = "date_understanding"

completion_metadata = CompletionMetadata(base_model=teacher_base_model, completion_key="zs_cot_test",
                                         dataset_key=dataset_key, prediction_template="zs_cot")
completion_dataset = infer_completion_data(completion_metadata, zs_cot_step=1,
                                           sample_indices=None, augs=1, temperature=0,
                                           max_tokens=128)


completion_dataset = infer_completion_data(completion_metadata, zs_cot_step=2,
                                           sample_indices=None, augs=1, temperature=0,
                                           max_tokens=128)


completion_identifier = CompletionIdentifier(teacher_base_model, "zs_cot_test", dataset_key)
completion_dataset = CompletionDataset.load(completion_identifier)


train, test = load_train_test_split(dataset_key)
evaluator = Evaluator.for_completion_dataset(completion_dataset)
evaluation = evaluator.evaluate_completion_dataset(completion_dataset, test)


print("Teacher model evaluation: {}".format(summarize_evaluation(evaluation)))


completion_identifier = CompletionIdentifier(teacher_base_model, "zs_cot_test", dataset_key)
completion_dataset = CompletionDataset.load(completion_identifier)
train, test = load_train_test_split(dataset_key)


finetune_key = "zs_cot_test_{}".format(dataset_key)
train_key = "ft_cot_test"


generate_finetune_data_from_completion_dataset(completion_dataset=completion_dataset,
                                               prediction_template="ft_cot_token",
                                               finetune_key=finetune_key,
                                               sample_indices=train,
                                               only_correct=True,  # default
                                              )


with open(get_finetune_data_path("openai", finetune_key)) as f:
    print(json.dumps(json.loads(f.readline()), indent=4))


init_finetune(finetune_key, base_model, dataset_key, train_key)
model_key = get_model_key(base_model, dataset_key, train_key)


completion_metadata = CompletionMetadata(base_model=base_model, completion_key="ft_cot_test",
                                         dataset_key=dataset_key, finetune_key=finetune_key,
                                         prediction_template="ft_cot_token",
                                         train_key=train_key, epoch=None)
train, test = load_train_test_split(dataset_key)
completion_dataset = infer_completion_data(completion_metadata, zs_cot_step=None,
                                           sample_indices=test, augs=1, temperature=0,
                                           max_tokens=1024)


completion_identifier = CompletionIdentifier(base_model, completion_key="ft_cot_test", dataset_key=dataset_key,
                                             train_key="ft_cot_test")
completion_dataset = CompletionDataset.load(completion_identifier)
train, test = load_train_test_split(dataset_key)


evaluator = Evaluator(dataset_key, "ft_cot_token")
evaluation = evaluator.evaluate_completion_dataset(completion_dataset, test)


print("student model evaluate: {}".format(summarize_evaluation(evaluation)))