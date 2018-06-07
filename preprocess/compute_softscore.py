#encoding=utf-8

#得到候选答案集合

#read:
# v2_mscoco_train2014_annotations.json v2_mscoco_val2014_annotations.json
# v2_OpenEnded_mscoco_train2014_questions.json v2_OpenEnded_mscoco_val2014_questions.json

#write:
# trainval_ans2label.pkl
# train_target.pkl val_target.pkl

import os
import json
import re
import pickle
import errno

# 缩略词表
contractions = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve":
        "could've", "couldnt": "couldn't", "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've", "didnt": "didn't", "doesnt":
        "doesn't", "dont": "don't", "hadnt": "hadn't", "hadnt've":
        "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent":
        "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve":
        "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll",
    "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im":
        "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've":
        "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's",
    "maam": "ma'am", "mightnt": "mightn't", "mightnt've":
        "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't",
    "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat":
        "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve":
        "she'd've", "she's": "she's", "shouldve": "should've", "shouldnt":
        "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve":
        "shouldn't've", "somebody'd": "somebodyd", "somebodyd've":
        "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll":
        "somebody'll", "somebodys": "somebody's", "someoned": "someone'd",
    "someoned've": "someone'd've", "someone'dve": "someone'd've",
    "someonell": "someone'll", "someones": "someone's", "somethingd":
        "something'd", "somethingd've": "something'd've", "something'dve":
        "something'd've", "somethingll": "something'll", "thats":
        "that's", "thered": "there'd", "thered've": "there'd've",
    "there'dve": "there'd've", "therere": "there're", "theres":
        "there's", "theyd": "they'd", "theyd've": "they'd've", "they'dve":
        "they'd've", "theyll": "they'll", "theyre": "they're", "theyve":
        "they've", "twas": "'twas", "wasnt": "wasn't", "wed've":
        "we'd've", "we'dve": "we'd've", "weve": "we've", "werent":
        "weren't", "whatll": "what'll", "whatre": "what're", "whats":
        "what's", "whatve": "what've", "whens": "when's", "whered":
        "where'd", "wheres": "where's", "whereve": "where've", "whod":
        "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl":
        "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll",
    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve":
        "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll":
        "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd":
        "you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll":
        "you'll", "youre": "you're", "youve": "you've"
}
manual_map = {'none': '0',
              'zero': '0',
              'one': '1',
              'two': '2',
              'three': '3',
              'four': '4',
              'five': '5',
              'six': '6',
              'seven': '7',
              'eight': '8',
              'nine': '9',
              'ten': '10'}
articles = ['a', 'an', 'the']
# <=匹配以开头的字符串;=匹配以结尾的字符串;?可选标志;！不含
# 去掉句号
period_strip = re.compile("(?!<=\d)(\.)(?!\d)")
# 去掉数字逗号
comma_strip = re.compile("(\d)(\,)(\d)")
# 斜杠为字符串中的特殊字符，加上r后变为原始字符串
punct = [';', r"/", '[', ']', '"', '{', '}',
         '(', ')', '=', '+', '\\', '_', '-',
         '>', '<', '@', '`', ',', '?', '!']

#创建目录
def create_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


def get_score(occurences):
    if occurences == 0:
        return 0
    elif occurences == 1:
        return 0.3
    elif occurences == 2:
        return 0.6
    elif occurences == 3:
        return 0.9
    else:
        return 1


# 处理标点符号
def process_punctuation(inText):
    outText = inText
    for p in punct:
        # 标点符号前后有空格
        if (p + ' ' in inText or ' ' + p in inText) \
                or (re.search(comma_strip, inText) != None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    # substitue替换 period_strip的模式替换空字符
    outText = period_strip.sub("", outText, re.UNICODE)
    return outText


def process_digit_article(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        # 如果键不存在于字典中，将会添加键并将值设为默认值
        word = manual_map.setdefault(word, word)
        if word not in articles:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = ' '.join(outText)
    return outText


def multiple_replace(text, wordDict):
    for key in wordDict:
        text = text.replace(key, wordDict[key])
    return text


# article冠词
def preprocess_doc(answer):
    answer = process_digit_article(process_punctuation(answer))
    answer = answer.replace(',', '')
    return answer


# 保留答案出现了最小次数以上的答案
def filter_answers(answers_dset, min_occurence):
    """This will change the answer to preprocessed version
    """
    occurence = {}

    for ans_entry in answers_dset:
        # 一个answer中的多个人的回答
        answers = ans_entry['answers']
        # 标准答案
        gtruth = ans_entry['multiple_choice_answer']
        # 去符号
        gtruth = preprocess_doc(gtruth)
        if gtruth not in occurence:
            occurence[gtruth] = set()
        occurence[gtruth].add(ans_entry['question_id'])
    # 整个答案集中答案出现次数少于约定次数的去掉
    for answer in list(occurence.keys()):
        if len(occurence[answer]) < min_occurence:
            occurence.pop(answer)

    print('Num of answers that appear >= %d times: %d' % (
        min_occurence, len(occurence)))
    return occurence


# 答案对应的下标
def create_ans2label(occurence, name, cache_root='maturedata'):
    """Note that this will also create label2ans.pkl at the same time

    occurence: dict {answer -> whatever}
    name: prefix of the output file
    cache_root: str
    """
    ans2label = {}
    label2ans = []
    label = 0
    for answer in occurence:
        # 答案的列表
        label2ans.append(answer)
        # 答案的序号
        ans2label[answer] = label
        label += 1

    create_dir(cache_root)

    cache_file = os.path.join(cache_root, name + '_ans2label.pkl')
    pickle.dump(ans2label, open(cache_file, 'wb'))
    cache_file = os.path.join(cache_root, name + '_label2ans.pkl')
    pickle.dump(label2ans, open(cache_file, 'wb'))
    return ans2label


def compute_target(answers_dset, ans2label, name, cache_root='maturedata'):
    """Augment answers_dset with soft score as label

    ***answers_dset should be preprocessed***

    Write result into a cache file
    """
    target = []
    for ans_entry in answers_dset:
        # answers列表
        answers = ans_entry['answers']
        answer_count = {}
        for answer in answers:
            # answer字典
            answer_ = answer['answer']
            # 统计每个答案的次数,key不存在返回0
            answer_count[answer_] = answer_count.get(answer_, 0) + 1

        labels = []
        scores = []
        for answer in answer_count:
            if answer not in ans2label:
                continue
            labels.append(ans2label[answer])
            score = get_score(answer_count[answer])
            scores.append(score)

        # 每张图和问题对应的答案位置和分数
        target.append({
            'question_id': ans_entry['question_id'],
            'image_id': ans_entry['image_id'],
            'labels': labels,
            'scores': scores,
            'type': ans_entry['answer_type']
        })

    create_dir(cache_root)
    cache_file = os.path.join(cache_root, name + '_target.pkl')
    pickle.dump(target, open(cache_file, 'wb'))
    return target


def get_answer(qid, answers):
    for ans in answers:
        if ans['question_id'] == qid:
            return ans


def get_question(qid, questions):
    for question in questions:
        if question['question_id'] == qid:
            return question


if __name__ == '__main__':
    # 答案
    train_answer_file = 'rawdata/v2_mscoco_train2014_annotations.json'
    train_answers = json.load(open(train_answer_file))['annotations']

    val_answer_file = 'rawdata/v2_mscoco_val2014_annotations.json'
    val_answers = json.load(open(val_answer_file))['annotations']

    # 问题
    train_question_file = 'rawdata/v2_OpenEnded_mscoco_train2014_questions.json'
    train_questions = json.load(open(train_question_file))['questions']

    val_question_file = 'rawdata/v2_OpenEnded_mscoco_val2014_questions.json'
    val_questions = json.load(open(val_question_file))['questions']

    answers = train_answers + val_answers
    occurence = filter_answers(answers, 9)
    print("filter done")
    ans2label = create_ans2label(occurence, 'trainval')
    print("answer label done")
    compute_target(train_answers, ans2label, 'train')
    compute_target(val_answers, ans2label, 'val')
    print("target done")

    #知识的预处理
    with open('maturedata/knowledge.json', 'r') as jf:
        knowldeges=json.load(jf)
    datas=[]
    for key in knowldeges.keys():
        item=knowldeges[key]
        datas.append(preprocess_doc(item))
    with open('maturedata/knowledge.json', 'w') as f:
        json.dump(datas,f)
