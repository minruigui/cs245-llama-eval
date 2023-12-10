import os
import numpy as np
import pandas as pd
import replicate
import random

from typing import List

os.environ["REPLICATE_API_TOKEN"] = "(...)"
choices = ["A", "B", "C", "D"]

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
    explanations = [
        'The sign of the covenant for Jewish males is circumcision. This practice is rooted in the covenant between God and Abraham, as described in the Hebrew Bible (Genesis 17). Circumcision involves the removal of the foreskin from the penis and is considered a significant religious and cultural rite in Judaism.',
        'The Three Jewels or Three Gems in Buddhism are: 1. Buddha (The Enlightened One) - The historical Buddha or the enlightened teacher., 2. Dharma (The Teaching) - The teachings of the Buddha. 3. Sangha (The Community) - The community of Buddhist monks and nuns.',
        'The concept of the "Mandate of Heaven" was developed during the Zhou Dynasty in ancient China. The Zhou rulers used this idea to legitimize their authority and justify the overthrow of the Shang Dynasty.',
        'Meiji Era.',
        'The Upanishads are primarily characterized as philosophical texts. They explore profound questions related to the nature of reality, the self (Atman), and the ultimate reality (Brahman). The Upanishads form the basis of Vedanta, one of the six orthodox schools of Hindu philosophy, and are revered for their philosophical and spiritual insights.'
    ]
    # explanations = [
    #     "The problem involves finding the least common multiple (LCM) of the blinking intervals for the red, yellow, and blue lights. The blinking intervals are: Red light blinks every 2 seconds. Yellow light blinks every 3 seconds. Blue light blinks every 5 seconds. The LCM of 2, 3, and 5 is 30 seconds. This means that every 30 seconds, all three lights will blink simultaneously. To find out how many times this happens in a 7-minute dance, convert 7 minutes to seconds (1 minute = 60 seconds): \(7 \text{ minutes} \times 60 \text{ seconds/minute} = 420 \text{ seconds}\) Now, divide the total time (420 seconds) by the LCM (30 seconds): \(420 \text{ seconds} / 30 \text{ seconds} = 14\) So, all three lights will come on at the same time 14 times during a 7-minute dance.",
    #     "The initial investment doubles in 6 years, so the annual interest rate (\(r\)) can be found using the formula: \[2 = \left(1 + \frac{r}{100}\right)^6\] Solving for \(r\), you find \(r = 100 \times (\sqrt[6]{2} - 1)\). Now, use \(r\) in the compound interest formula to find the time (\(t\)) it takes for $300 to grow to $9600: \[9600 = 300 \left(1 + \frac{r}{100}\right)^t\] Solve for \(t\).",
    #     "Given that \(x\) varies directly as the square of \(y\) and \(y\) varies directly as the cube of \(z\), we have the relationships: 1. \(x = k_1y^2\) 2. \(y = k_2z^3\) Combining these, we get \(x = k_1(k_2z^3)^2\). When \(z = 2\), and \(x = -16\): \[-16 = k_1(k_2 \cdot 2^3)^2\] This simplifies to \(-16 = k_1 \cdot k_2^2 \cdot 64\). Let \(C = k_1 \cdot k_2^2\). Now, when \(z = \frac{1}{2}\): \[x = C \cdot \left(\frac{1}{8}\right)^2\] Substitute the value of \(C\): \[x = (-16) \cdot \left(\frac{1}{8}\right)^2\] This simplifies to \(x = -\frac{1}{4}\). Therefore, when \(z = \frac{1}{2}\), the value of \(x\) is \(-\frac{1}{4}\).",
    #     "Let's simplify the expression step by step: 1. Start with the innermost part: \(\frac{1}{729}\) inside the square root. 2. Take the square root of \(\frac{1}{729}\): \(\sqrt{\frac{1}{729}} = \frac{1}{27}\). 3. Take the cube root of \(\frac{1}{27}\): \(\sqrt[3]{\frac{1}{27}} = \frac{1}{3}\). 4. Finally, take the square root of \(\frac{1}{3}\): \(\sqrt{\frac{1}{3}}\). To write the result with a rational denominator, multiply the numerator and denominator by \(\sqrt{3}\) to rationalize the denominator: \[\sqrt{\frac{1}{3}} \times \frac{\sqrt{3}}{\sqrt{3}} = \frac{\sqrt{3}}{3}\] So, \(\sqrt{\sqrt[3]{\sqrt{\frac{1}{729}}}} = \frac{\sqrt{3}}{3}\).",
    #     "To find the mean (average) of the test scores, you sum up all the scores and then divide by the number of students. \[ \text{Mean} = \frac{45 + 55 + 50 + 70 + 65 + 80 + 40 + 90 + 70 + 85}{10} \] Adding up the scores: \(45 + 55 + 50 + 70 + 65 + 80 + 40 + 90 + 70 + 85 = 680\) Now, divide by the number of students (10): \[ \text{Mean} = \frac{680}{10} = 68 \] Therefore, the mean of the students' test scores is 68."
    # ]
    prompt = '{\n\t"Question": "' + df.iloc[idx, 0] + '",\n\t"Answer choices": ['
    k = df.shape[1] - 2
    answers = {}
    for j in range(k):
        prompt += f'\n\t\t\"{choices[j]}) {df.iloc[idx, j+1]}",'
        answers[choices[j]] = df.iloc[idx, j+1]
    prompt = prompt[:-1] + "\n\t]"
    if include_answer:
        prompt += f',\n\t"Explanation": "{explanations[idx]}", \n\t"Answer": "{df.iloc[idx, k + 1]}) {answers[df.iloc[idx, k + 1]]}"\n}}\n\n'
    else:
        prompt += '\n{\n\t"Explanation": "'
    return prompt

def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {} with JSON responses. Please format your responses in JSON. Please keep it concise.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt
BATCH_SIZE = 1

class Replicate():
    def __init__(self):
        pass

    def text_completion(self, prompts, max_gen_len, temperature, top_p, logprobs):
        for prompt in prompts:
            output = replicate.run(
                "meta/llama-2-7b:77dde5d6c56598691b9008f7d123a18d98f40e4b4978f8a72215ebfc2553ddd8",
                input={
                    "prompt": prompt,
                    'max_length': 10_000,
                    'temperature': 0.01,
                    'top_p': 0.9
                }
            )

            output = ''.join(list(output))

            if '"Answer":' in output:
                try:
                    result = output.split('"Answer": "')[1]
                    if ")" in result:
                        result = result.split(')')[0].strip()
                        return result
                except:
                    return "X"
                
            return "X"


generator = Replicate()

def eval(args, subject, engine, dev_df, test_df):
    cors = []
    all_probs = []
    answers = choices[:test_df.shape[1]-2]
    prompts = []
    labels=[]
    for i in range(test_df.shape[0]):
        print(f'{i+1} / {test_df.shape[0]}', end='')
        if random.random() > 0.25:
            print('...randomly skipping.')
            continue
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end
        label = test_df.iloc[i, test_df.shape[1]-1]
        prompts.append(prompt)
        labels.append(label)
        if len(prompts) == BATCH_SIZE:
            cs = generator.text_completion(
                prompts,
                max_gen_len=1,
                temperature=0,
                top_p=0.9,
                logprobs=True
            )
            for c,l in zip(cs,labels):
                print('...', c, l)
                cors.append(c==l)
            prompts=[]
            labels=[]

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

        for subject in subjects:
            if subject not in [
                # 'high_school_mathematics'
                'world_religions'
                ]:
                continue

            dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None)[:args.ntrain]
            test_df = pd.read_csv(os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None)


            cors, acc, probs = eval(args, subject, engine, dev_df, test_df)
            all_cors.append(cors)

            test_df["{}_correct".format(engine)] = cors
            test_df.to_csv(os.path.join(args.save_dir, "results_{}".format(engine), "{}.csv".format(subject)), index=None)

        weighted_acc = np.mean(np.concatenate(all_cors))
        print("Average accuracy: {:.3f}".format(weighted_acc))

if __name__ == "__main__":
    from types import SimpleNamespace
    main(SimpleNamespace(**{"ntrain":5,"data_dir":"data","save_dir":"results","engine":["davinci"]}))

