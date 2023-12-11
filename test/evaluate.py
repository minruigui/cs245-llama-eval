# import argparse
# import openai
# import os
# import numpy as np
# import pandas as pd
# import time
# import fire

# from llama import Llama
# from typing import List

# from llama.tokenizer import Tokenizer
# openai.api_key = "xxxx"
# choices = ["A", "B", "C", "D"]


# def softmax(x):
#     z = x - max(x)
#     numerator = np.exp(z)
#     denominator = np.sum(numerator)
#     softmax = numerator/denominator
#     return softmax

# def format_subject(subject):
#     l = subject.split("_")
#     s = ""
#     for entry in l:
#         s += " " + entry
#     return s

# def format_example(df, idx, include_answer=True):
#     prompt = df.iloc[idx, 0]
#     k = df.shape[1] - 2
#     for j in range(k):
#         prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
#     prompt += "\nAnswer:"
#     if include_answer:
#         prompt += " {}\n\n".format(df.iloc[idx, k + 1])
#     return prompt

# def gen_prompt(train_df, subject, k=-1):
#     prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
#     if k == -1:
#         k = train_df.shape[0]
#     for i in range(k):
#         prompt += format_example(train_df, i)
#     return prompt
# BACH_SIZE = 1
# # generator = Llama.build(
# # ckpt_dir='/home/ubuntu/projects/llama/llama-2-7b',
# # tokenizer_path='/home/ubuntu/projects/llama/tokenizer.model',
# # max_seq_len=4096,
# # max_batch_size=BACH_SIZE,
# # )


# def ape_prompt_gen(df, idx, include_answer=True):
#     question = df.iloc[idx, 0]
#     k = df.shape[1] - 2
#     answer = df.iloc[idx, k + 1]
#     eval_template = \
#         """Instruction: [PROMPT]
#     Input: [INPUT]
#     Output: [OUTPUT]"""
#     from automatic_prompt_engineer import ape
#     result, demo_fn = ape.simple_ape(
#         dataset=(question, answer),
#         eval_template=eval_template,
#     )
#     return result[0][result[0].index(": "):]

# def eval(args, subject, engine, dev_df, test_df):
#     tokenizer = Tokenizer(model_path='/home/ubuntu/projects/llama/tokenizer.model')
#     def crop_prompt(prompt: str):
#         cropped_prompt = tokenizer.decode(tokenizer.encode(prompt,bos=True, eos=False)[:2048])
#         return cropped_prompt

#     def crop(s):
#         prompt = crop_prompt(s)
#         return prompt
#     cors = []
#     all_probs = []
#     answers = choices[:test_df.shape[1]-2]
#     prompts = []
#     labels=[]
#     for i in range(test_df.shape[0]):
#         # get prompt and make sure it fits
#         k = args.ntrain
#         prompt_end = format_example(test_df, i, include_answer=False)
#         ape_prompt = ape_prompt_gen(dev_df, i, include_answer=False)
#         train_prompt = gen_prompt(dev_df, subject, k)
#         prompt = train_prompt + ape_prompt + prompt_end

#         while crop(prompt) != prompt:
#             k -= 1
#             train_prompt = gen_prompt(dev_df, subject, k)
#             prompt = train_prompt + prompt_end

#         label = test_df.iloc[i, test_df.shape[1]-1]


#                 # c = openai.Completion.create(
#                 #     engine=engine,
#                 #     prompt=prompt,
#                 #     max_tokens=1,
#                 #     logprobs=100,
#                 #     temperature=0,
#                 #     echo=True
#                 # )
#         prompts.append(prompt)
#         labels.append(label)
#         if len(prompts) == BACH_SIZE:
#             cs = generator.text_completion(
#                 prompts,
#                 max_gen_len=1,
#                 temperature=0,
#                 top_p=0.9,
#                 logprobs=True
#             )
#             for c,l in zip(cs,labels):
#                 cors.append(c['generation']==l)
#             prompts=[]
#             labels=[]

#         # lprobs = []
#         # for ans in answers:
#         #     try:
#         #         lprobs.append(c["choices"][0]["logprobs"]["top_logprobs"][-1][" {}".format(ans)])
#         #     except:
#         #         print("Warning: {} not found. Artificially adding log prob of -100.".format(ans))
#         #         lprobs.append(-100)
#         # pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(lprobs)]
#         # probs = softmax(np.array(lprobs))

#         # cor = pred == label
#         # cors.append(cor)
#         # all_probs.append(probs)
#     if len(prompts)>0:
#         cs = generator.text_completion(
#                 prompts,
#                 max_gen_len=1,
#                 temperature=0,
#                 top_p=0.9,
#                 logprobs=True
#             )
#         for c,l in zip(cs,labels):
#             cors.append(c['generation']==l)

#     acc = np.mean(cors)
#     cors = np.array(cors)

#     all_probs = np.array(all_probs)
#     print("Average accuracy {:.3f} - {}".format(acc, subject))

#     return cors, acc, all_probs

# def main(args):
#     engines = args.engine
#     subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(args.data_dir, "test")) if "_test.csv" in f])

#     if not os.path.exists(args.save_dir):
#         os.mkdir(args.save_dir)
#     for engine in engines:
#         if not os.path.exists(os.path.join(args.save_dir, "results_{}".format(engine))):
#             os.mkdir(os.path.join(args.save_dir, "results_{}".format(engine)))

#     print(subjects)
#     print(args)

#     for engine in engines:
#         print(engine)
#         all_cors = []

#         for subject in subjects:
#             dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None)[:args.ntrain]
#             test_df = pd.read_csv(os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None)

#             cors, acc, probs = eval(args, subject, engine, dev_df, test_df)
#             all_cors.append(cors)

#             test_df["{}_correct".format(engine)] = cors
#             # for j in range(probs.shape[1]):
#             #     choice = choices[j]
#             #     test_df["{}_choice{}_probs".format(engine, choice)] = probs[:, j]
#             test_df.to_csv(os.path.join(args.save_dir, "results_{}".format(engine), "{}.csv".format(subject)), index=None)

#         weighted_acc = np.mean(np.concatenate(all_cors))
#         print("Average accuracy: {:.3f}".format(weighted_acc))

# if __name__ == "__main__":
#     # parser = argparse.ArgumentParser()
#     # parser.add_argument("--ntrain", "-k", type=int, default=5)
#     # parser.add_argument("--data_dir", "-d", type=str, default="data")
#     # parser.add_argument("--save_dir", "-s", type=str, default="results")
#     # parser.add_argument("--engine", "-e", choices=["davinci", "curie", "babbage", "ada","gpt4"],
#     #                     default=["davinci", "curie", "babbage", "ada"], nargs="+")
#     # args = parser.parse_args()
#     from types import SimpleNamespace
#     main(SimpleNamespace(**{"ntrain":5,"data_dir":"data","save_dir":"results","engine":["davinci"]}))


import argparse
import openai
import os
import numpy as np
import pandas as pd
import time
import fire
import replicate
# from llama import Llama
from typing import List

# from llama.tokenizer import Tokenizer
openai.api_key = "xxxx"
choices = ["A", "B", "C", "D"]
api = replicate.Client(api_token="xxxx")



def softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator/denominator
    return softmax

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    d = {}
    rd = {}
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
        d[choices[j]] = df.iloc[idx, j+1]
        rd[df.iloc[idx, j+1]] = choices[j]
    prompt += "\nAnswer:"
    if include_answer:
        prompt += "{}\n\n".format(df.iloc[idx, k+1])
    return prompt

def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt
BACH_SIZE = 1
# generator = Llama.build(
# ckpt_dir='/home/ubuntu/projects/llama/llama-2-7b',
# tokenizer_path='/home/ubuntu/projects/llama/tokenizer.model',
# max_seq_len=4096,
# max_batch_size=BACH_SIZE,
# )


def get_q_a(df):
    k = df.shape[0]
    questions = []
    answers = []
    d = {}
    rd = {}
    for i in range(k):
      questions.append(df.iloc[i, 0])
    for i in range(k):
      for j in range(df.shape[1] - 2):
        d[choices[j]] = df.iloc[i, j+1]
        rd[df.iloc[i, j+1]] = choices[j]
      answers.append("{}\n\n".format(rd[d[df.iloc[i, df.shape[1] - 1]]]))
    return questions, answers


def ape_prompt_gen(df):
    questions, answers = get_q_a(df)
    eval_template = \
    """Instruction: [PROMPT]
    Input: [INPUT]
    Output: [OUTPUT]"""
    from automatic_prompt_engineer import ape
    result, demo_fn = ape.simple_ape(
        dataset=(questions, answers),
        eval_template=eval_template,
        eval_model='text-davinci-003',
        prompt_gen_model='text-davinci-003',
        num_prompts=10
    )
    return result.prompts[0]

def eval(args, subject, engine, dev_df, test_df):
    # tokenizer = Tokenizer(model_path='/home/ubuntu/projects/llama/tokenizer.model')
    # def crop_prompt(prompt: str):
    #     cropped_prompt = tokenizer.decode(tokenizer.encode(prompt,bos=True, eos=False)[:2048])
    #     return cropped_prompt

    # def crop(s):
    #     prompt = crop_prompt(s)
    #     return prompt
    cors = []
    all_probs = []
    answers = choices[:test_df.shape[1]-2]
    prompts = []
    labels=[]
    model_labels=[]
    ape_prompt ="Act as an expert in {} ".format(subject) + ape_prompt_gen(dev_df)
    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        # print(prompt_end)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + ape_prompt + prompt_end
        print(prompt)
        label = test_df.iloc[i, test_df.shape[1]-1]
        prompts.append(prompt)
        labels.append(label)
        if len(prompts) == BACH_SIZE:
            cs = api.run(
                "meta/llama-2-7b:77dde5d6c56598691b9008f7d123a18d98f40e4b4978f8a72215ebfc2553ddd8",
                input={"prompt": prompt, "top_p": 0.9, "temperature": 0.01}
            )
            output = " ".join(list(cs))
            ans_idx = output.find(next(filter(str.isalpha, output)))
            print("_________{}__________".format(output[ans_idx]))
            model_labels.append(output[ans_idx])
            for c,l in zip(model_labels,labels):
                cors.append(c==l)
            prompts=[]
            labels=[]
            model_labels = []
    if len(prompts)>0:
        cs = api.run(
                "meta/llama-2-7b:77dde5d6c56598691b9008f7d123a18d98f40e4b4978f8a72215ebfc2553ddd8",
                input={"prompt": prompt, "top_p": 0.9, "temperature": 0.01}
            )
        output = " ".join(list(cs))
        model_label = output.find(next(filter(str.isalpha, output)))
        model_labels.append(model_label)
        for c,l in zip(model_labels,labels):
          cors.append(c==l)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    with open('results/{}_res.txt'.format(subject), 'w') as f:
      f.write("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs

def main(args):
    engines = args.engine
    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(args.data_dir, "test")) if "_test.csv" in f])

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    for engine in engines:
        if not os.path.exists(os.path.join(args.save_dir, "results_{}".format(engine))):
            os.mkdir(os.path.join(args.save_dir, "results_{}".format(engine)))

    print(subjects)
    print(args)

    for engine in engines:
        print(engine)
        all_cors = []

        for subject in ["high_school_mathematics", "world_religions"]:
            dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None)[:args.ntrain]
            test_df = pd.read_csv(os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None)

            cors, acc, probs = eval(args, subject, engine, dev_df, test_df)
            all_cors.append(cors)

            test_df["{}_correct".format(engine)] = cors
            # for j in range(probs.shape[1]):
            #     choice = choices[j]
            #     test_df["{}_choice{}_probs".format(engine, choice)] = probs[:, j]
            test_df.to_csv(os.path.join(args.save_dir, "results_{}".format(engine), "{}.csv".format(subject)), index=None)

        weighted_acc = np.mean(np.concatenate(all_cors))
        print("Average accuracy: {:.3f}".format(weighted_acc))

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--ntrain", "-k", type=int, default=5)
    # parser.add_argument("--data_dir", "-d", type=str, default="data")
    # parser.add_argument("--save_dir", "-s", type=str, default="results") 
    # parser.add_argument("--engine", "-e", choices=["davinci", "curie", "babbage", "ada","gpt4"],
    #                     default=["davinci", "curie", "babbage", "ada"], nargs="+")
    # args = parser.parse_args()
    from types import SimpleNamespace
    main(SimpleNamespace(**{"ntrain":5,"data_dir":"data","save_dir":"results","engine":["davinci"]}))

# import argparse
# import openai
# import os
# import numpy as np
# import pandas as pd
# import time
# import fire

# from llama import Llama
# from typing import List

# from llama.tokenizer import Tokenizer
# openai.api_key = "xxxx"
# choices = ["A", "B", "C", "D"]


# def softmax(x):
#     z = x - max(x)
#     numerator = np.exp(z)
#     denominator = np.sum(numerator)
#     softmax = numerator/denominator
#     return softmax

# def format_subject(subject):
#     l = subject.split("_")
#     s = ""
#     for entry in l:
#         s += " " + entry
#     return s

# def format_example(df, idx, include_answer=True):
#     prompt = df.iloc[idx, 0]
#     k = df.shape[1] - 2
#     for j in range(k):
#         prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
#     prompt += "\nAnswer:"
#     if include_answer:
#         prompt += " {}\n\n".format(df.iloc[idx, k + 1])
#     return prompt

# def gen_prompt(train_df, subject, k=-1):
#     prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
#     if k == -1:
#         k = train_df.shape[0]
#     for i in range(k):
#         prompt += format_example(train_df, i)
#     return prompt
# BACH_SIZE = 1
# # generator = Llama.build(
# # ckpt_dir='/home/ubuntu/projects/llama/llama-2-7b',
# # tokenizer_path='/home/ubuntu/projects/llama/tokenizer.model',
# # max_seq_len=4096,
# # max_batch_size=BACH_SIZE,
# # )


# def ape_prompt_gen(df, idx, include_answer=True):
#     question = df.iloc[idx, 0]
#     k = df.shape[1] - 2
#     answer = df.iloc[idx, k + 1]
#     eval_template = \
#         """Instruction: [PROMPT]
#     Input: [INPUT]
#     Output: [OUTPUT]"""
#     from automatic_prompt_engineer import ape
#     result, demo_fn = ape.simple_ape(
#         dataset=(question, answer),
#         eval_template=eval_template,
#     )
#     return result[0][result[0].index(": "):]

# def eval(args, subject, engine, dev_df, test_df):
#     tokenizer = Tokenizer(model_path='/home/ubuntu/projects/llama/tokenizer.model')
#     def crop_prompt(prompt: str):
#         cropped_prompt = tokenizer.decode(tokenizer.encode(prompt,bos=True, eos=False)[:2048])
#         return cropped_prompt

#     def crop(s):
#         prompt = crop_prompt(s)
#         return prompt
#     cors = []
#     all_probs = []
#     answers = choices[:test_df.shape[1]-2]
#     prompts = []
#     labels=[]
#     for i in range(test_df.shape[0]):
#         # get prompt and make sure it fits
#         k = args.ntrain
#         prompt_end = format_example(test_df, i, include_answer=False)
#         ape_prompt = ape_prompt_gen(dev_df, i, include_answer=False)
#         train_prompt = gen_prompt(dev_df, subject, k)
#         prompt = train_prompt + ape_prompt + prompt_end

#         while crop(prompt) != prompt:
#             k -= 1
#             train_prompt = gen_prompt(dev_df, subject, k)
#             prompt = train_prompt + prompt_end

#         label = test_df.iloc[i, test_df.shape[1]-1]


#                 # c = openai.Completion.create(
#                 #     engine=engine,
#                 #     prompt=prompt,
#                 #     max_tokens=1,
#                 #     logprobs=100,
#                 #     temperature=0,
#                 #     echo=True
#                 # )
#         prompts.append(prompt)
#         labels.append(label)
#         if len(prompts) == BACH_SIZE:
#             cs = generator.text_completion(
#                 prompts,
#                 max_gen_len=1,
#                 temperature=0,
#                 top_p=0.9,
#                 logprobs=True
#             )
#             for c,l in zip(cs,labels):
#                 cors.append(c['generation']==l)
#             prompts=[]
#             labels=[]

#         # lprobs = []
#         # for ans in answers:
#         #     try:
#         #         lprobs.append(c["choices"][0]["logprobs"]["top_logprobs"][-1][" {}".format(ans)])
#         #     except:
#         #         print("Warning: {} not found. Artificially adding log prob of -100.".format(ans))
#         #         lprobs.append(-100)
#         # pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(lprobs)]
#         # probs = softmax(np.array(lprobs))

#         # cor = pred == label
#         # cors.append(cor)
#         # all_probs.append(probs)
#     if len(prompts)>0:
#         cs = generator.text_completion(
#                 prompts,
#                 max_gen_len=1,
#                 temperature=0,
#                 top_p=0.9,
#                 logprobs=True
#             )
#         for c,l in zip(cs,labels):
#             cors.append(c['generation']==l)

#     acc = np.mean(cors)
#     cors = np.array(cors)

#     all_probs = np.array(all_probs)
#     print("Average accuracy {:.3f} - {}".format(acc, subject))

#     return cors, acc, all_probs

# def main(args):
#     engines = args.engine
#     subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(args.data_dir, "test")) if "_test.csv" in f])

#     if not os.path.exists(args.save_dir):
#         os.mkdir(args.save_dir)
#     for engine in engines:
#         if not os.path.exists(os.path.join(args.save_dir, "results_{}".format(engine))):
#             os.mkdir(os.path.join(args.save_dir, "results_{}".format(engine)))

#     print(subjects)
#     print(args)

#     for engine in engines:
#         print(engine)
#         all_cors = []

#         for subject in subjects:
#             dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None)[:args.ntrain]
#             test_df = pd.read_csv(os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None)

#             cors, acc, probs = eval(args, subject, engine, dev_df, test_df)
#             all_cors.append(cors)

#             test_df["{}_correct".format(engine)] = cors
#             # for j in range(probs.shape[1]):
#             #     choice = choices[j]
#             #     test_df["{}_choice{}_probs".format(engine, choice)] = probs[:, j]
#             test_df.to_csv(os.path.join(args.save_dir, "results_{}".format(engine), "{}.csv".format(subject)), index=None)

#         weighted_acc = np.mean(np.concatenate(all_cors))
#         print("Average accuracy: {:.3f}".format(weighted_acc))

# if __name__ == "__main__":
#     # parser = argparse.ArgumentParser()
#     # parser.add_argument("--ntrain", "-k", type=int, default=5)
#     # parser.add_argument("--data_dir", "-d", type=str, default="data")
#     # parser.add_argument("--save_dir", "-s", type=str, default="results")
#     # parser.add_argument("--engine", "-e", choices=["davinci", "curie", "babbage", "ada","gpt4"],
#     #                     default=["davinci", "curie", "babbage", "ada"], nargs="+")
#     # args = parser.parse_args()
#     from types import SimpleNamespace
#     main(SimpleNamespace(**{"ntrain":5,"data_dir":"data","save_dir":"results","engine":["davinci"]}))


import argparse
import openai
import os
import numpy as np
import pandas as pd
import time
import fire
import replicate
# from llama import Llama
from typing import List

# from llama.tokenizer import Tokenizer
openai.api_key = "xxxxx"
choices = ["A", "B", "C", "D"]
api = replicate.Client(api_token="xxxx")



def softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator/denominator
    return softmax

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    d = {}
    rd = {}
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
        d[choices[j]] = df.iloc[idx, j+1]
        rd[df.iloc[idx, j+1]] = choices[j]
    prompt += "\nAnswer:"
    if include_answer:
        prompt += "The answer to \"{}\" is {} which is {}\n\n".format(
            df.iloc[idx, 0], d[df.iloc[idx, k+1]], rd[d[df.iloc[idx, k+1]]])
    return prompt

def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt
BACH_SIZE = 1
# generator = Llama.build(
# ckpt_dir='/home/ubuntu/projects/llama/llama-2-7b',
# tokenizer_path='/home/ubuntu/projects/llama/tokenizer.model',
# max_seq_len=4096,
# max_batch_size=BACH_SIZE,
# )


def get_q_a(df):
    k = df.shape[0]
    questions = []
    answers = []
    d = {}
    rd = {}
    for i in range(k):
      questions.append(df.iloc[i, 0])
    for i in range(k):
      for j in range(df.shape[1] - 2):
        d[choices[j]] = df.iloc[i, j+1]
        rd[df.iloc[i, j+1]] = choices[j]
      answers.append("The answer to \"{}\" is {} which is {}\n\n".format(
          df.iloc[i, 0], d[df.iloc[i, df.shape[1] - 1]], rd[d[df.iloc[i, df.shape[1] - 1]]]))
    return questions, answers


def ape_prompt_gen(df):
    questions, answers = get_q_a(df)
    eval_template = \
    """Instruction: [PROMPT]
    Input: [INPUT]
    Output: [OUTPUT]"""
    from automatic_prompt_engineer import ape
    result, demo_fn = ape.simple_ape(
        dataset=(questions[:5], answers[:5]),
        eval_template=eval_template,
        eval_model='text-davinci-002',
        prompt_gen_model='text-davinci-002',
        num_prompts=10
    )
    return result.prompts[0]

def eval(args, subject, engine, dev_df, test_df):
    # tokenizer = Tokenizer(model_path='/home/ubuntu/projects/llama/tokenizer.model')
    # def crop_prompt(prompt: str):
    #     cropped_prompt = tokenizer.decode(tokenizer.encode(prompt,bos=True, eos=False)[:2048])
    #     return cropped_prompt

    # def crop(s):
    #     prompt = crop_prompt(s)
    #     return prompt
    cors = []
    all_probs = []
    answers = choices[:test_df.shape[1]-2]
    prompts = []
    labels=[]
    ape_prompt ="Based on examples above " + ape_prompt_gen(dev_df) + " including intermediate steps"
    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        # print(prompt_end)
        train_prompt = gen_prompt(dev_df, subject, k)
        print(train_prompt)
        prompt = train_prompt + ape_prompt + prompt_end
        print(prompt)
        # while crop(prompt) != prompt:
        #     k -= 1
        #     train_prompt = gen_prompt(dev_df, subject, k)
        #     prompt = train_prompt + prompt_end

        label = test_df.iloc[i, test_df.shape[1]-1]


                # c = openai.Completion.create(
                #     engine=engine,
                #     prompt=prompt,
                #     max_tokens=1,
                #     logprobs=100,
                #     temperature=0,
                #     echo=True
                # )
        prompts.append(prompt)
        labels.append(label)
        if len(prompts) == BACH_SIZE:
            # cs = generator.text_completion(
            #     prompts,
            #     max_gen_len=1,
            #     temperature=0,
            #     top_p=0.9,
            #     logprobs=True
            # )
            cs = api.run(
                "meta/llama-2-7b:77dde5d6c56598691b9008f7d123a18d98f40e4b4978f8a72215ebfc2553ddd8",
                input={"prompt": prompt}
            )
            for output in cs:
              print(output, end="")
            for c,l in zip(cs,labels):
                cors.append(c['generation']==l)
            prompts=[]
            labels=[]

        # lprobs = []
        # for ans in answers:
        #     try:
        #         lprobs.append(c["choices"][0]["logprobs"]["top_logprobs"][-1][" {}".format(ans)])
        #     except:
        #         print("Warning: {} not found. Artificially adding log prob of -100.".format(ans))
        #         lprobs.append(-100)
        # pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(lprobs)]
        # probs = softmax(np.array(lprobs))

        # cor = pred == label
        # cors.append(cor)
        # all_probs.append(probs)
    if len(prompts)>0:
        cs = api.run(
                "meta/llama-2-7b:77dde5d6c56598691b9008f7d123a18d98f40e4b4978f8a72215ebfc2553ddd8",
                input={"prompt": prompt}
            )
        for output in cs:
          print(output, end="")
        for c,l in zip(cs,labels):
          cors.append(c['generation']==l)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs

def main(args):
    engines = args.engine
    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(args.data_dir, "test")) if "_test.csv" in f])

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    for engine in engines:
        if not os.path.exists(os.path.join(args.save_dir, "results_{}".format(engine))):
            os.mkdir(os.path.join(args.save_dir, "results_{}".format(engine)))

    print(subjects)
    print(args)

    for engine in engines:
        print(engine)
        all_cors = []

        for subject in ["high_school_mathematics", "world_religions"]:
            dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None)[:args.ntrain]
            test_df = pd.read_csv(os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None)

            cors, acc, probs = eval(args, subject, engine, dev_df, test_df)
            all_cors.append(cors)

            test_df["{}_correct".format(engine)] = cors
            # for j in range(probs.shape[1]):
            #     choice = choices[j]
            #     test_df["{}_choice{}_probs".format(engine, choice)] = probs[:, j]
            test_df.to_csv(os.path.join(args.save_dir, "results_{}".format(engine), "{}.csv".format(subject)), index=None)

        weighted_acc = np.mean(np.concatenate(all_cors))
        print("Average accuracy: {:.3f}".format(weighted_acc))

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--ntrain", "-k", type=int, default=5)
    # parser.add_argument("--data_dir", "-d", type=str, default="data")
    # parser.add_argument("--save_dir", "-s", type=str, default="results") 
    # parser.add_argument("--engine", "-e", choices=["davinci", "curie", "babbage", "ada","gpt4"],
    #                     default=["davinci", "curie", "babbage", "ada"], nargs="+")
    # args = parser.parse_args()
    from types import SimpleNamespace
    main(SimpleNamespace(**{"ntrain":5,"data_dir":"data","save_dir":"results","engine":["davinci"]}))

