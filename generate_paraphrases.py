import argparse,sys
import random
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
import rstr
import json
import pandas as pd
import spacy
from spacy.matcher import Matcher


class Paraphrase:
    def __init__(self, construction_data, transformation_type, label, text):
        self.label = label
        self.transformation_type = transformation_type
        self.text = text
        self.construction_data = construction_data

    def get_string(self):
        out = ": ".join(
            [self.label, self.transformation_type.__name__.upper()]) + ": "
        data = ""
        if self.construction_data:
            data = self.construction_data + " > "
        out += data
        out += self.text
        return out


class Paraphrase_Maker:
    nlp = spacy.load("en_core_web_sm")
    existing_texts = set([])
    existing_paraphrases = set([])
    non_seeds_counter = {}
    label2paraphrases = {}
    df = pd.DataFrame(columns = ["label","text","transformation_type","construction_data"])

    def __init__(self, regex_tsv, general_rules_tsv, seeds_tsv, pattern_json):
        # open regexes
        self.label_regex_pairs = self.tsv_opener(regex_tsv)

        # open general rules
        self.grule_replacement_pairs = self.tsv_opener(general_rules_tsv)

        # open seeds
        label_seed_pairs = self.tsv_opener(seeds_tsv)
        for (label, seed) in label_seed_pairs:
            seed_paraphrase = Paraphrase(None, self.seed, label, seed)
            self.non_seeds_counter[label] = 0
            if label in self.label2paraphrases:
                self.label2paraphrases[label].append(seed_paraphrase)
            else:
                self.label2paraphrases[label] = [seed_paraphrase]
            self.paraphrase_adder(seed_paraphrase)

        # open chunking rules
        self.chunk_matcher = self.get_pattern_matcher(pattern_json)

    def tsv_opener(self, path_):
        out = set([])
        with open(path_) as x:
            for line in x:
                [a, b] = re.split(" *\t *", line)
                b = self.norm(b)
                out.add((a, b))
        return out

    def seed(self):
        pass

    def paraphrase_adder(self, p):
        self.existing_texts.add(p.text)
        self.existing_paraphrases.add(p)
        if p.transformation_type is not None:
            self.non_seeds_counter[p.label] += 1
        self.label2paraphrases[p.label].append(p)
        self.df.loc[len(self.df.index)] = \
        {"label":p.label,"text":p.text,"transformation_type":p.transformation_type.__name__,"construction_data":p.construction_data}

    def norm(self, s):
        out = s.strip().lower().rstrip(".?!")
        out = re.sub(" +", " ", out)
        return out

    def get_pattern_matcher(self, pattern_json):
        with open(pattern_json) as pj:
            patterns = json.load(pj)
        matcher = Matcher(self.nlp.vocab)
        for pattern in patterns:
            matcher.add("", [pattern])
        return matcher

    def lemmatize_a_word(self, p):
        s = p.text
        toks = nltk.word_tokenize(s)
        pos = nltk.pos_tag(toks)
        i_change = random.choice(range(0, len(pos)))
        i_looks = 0
        found_lem = True
        wordnet_lemmatizer = WordNetLemmatizer()
        while pos[i_change][1] not in ["VBZ", "VBG", "NNS", "VBD"] \
        or toks[i_change] in ["being", "is", "are", "were", "was", "am", "savings"]:
            i_change = random.choice(range(0, len(pos)))
            i_looks += 1
            if i_looks > 10:
                found_lem = False
                break
        if found_lem:
            toks[i_change] = wordnet_lemmatizer.lemmatize(
                toks[i_change], pos=pos[i_change][1].lower()[0])
        detokenizer = TreebankWordDetokenizer()
        return Paraphrase(s, self.lemmatize_a_word, p.label, detokenizer.detokenize(toks).replace(" .", "."))

    def replace_from_general_rules(self, p):
        s = p.text[:]
        (k, v) = random.sample(self.grule_replacement_pairs, 1)[0]
        v = random.choice(v.split("||"))
        if re.findall(k, s):
            s = re.sub(k, v, s)
        return Paraphrase(k, self.replace_from_general_rules, p.label, self.norm(s))

    def generate_from_regex(self, p):
        label = p.label
        while True:
            random_rule = random.sample(self.label_regex_pairs, 1)[0]
            if random_rule[0] == label:
                break
        # return self.norm(rstr.xeger(random_rule[1]))
        return Paraphrase(None, self.generate_from_regex, label, self.norm(rstr.xeger(random_rule[1])))

    def get_pattern_chunks(self, p):
        doc = self.nlp(p.text)
        matches = self.chunk_matcher(doc)
        span = None
        for match_id, start, end in matches:
            span = doc[start:end]  # The matched span
            # print(span.text)
        if span:
            return Paraphrase(p.text, self.get_pattern_chunks, p.label, span.text)
        return p

    def create_paraphrase(self, paraphrase_in, transformer):
        return transformer(paraphrase_in)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--regex_tsv', help='path to regex file', required=True)
    parser.add_argument('--general_rules_tsv',
                        help='path to general rules', required=True)
    parser.add_argument('--seeds_tsv', help='path to seeds', required=True)
    parser.add_argument(
        '--end_number', help='list of possible string replacements', required=True)
    parser.add_argument('--patterns_json',
                        help='json of spacy patterns', required=True)
    args = parser.parse_args()
    return args


args = parse_arguments()


paraphrase_maker = Paraphrase_Maker(
    args.regex_tsv, args.general_rules_tsv, args.seeds_tsv, args.patterns_json)
select_labels = list(paraphrase_maker.non_seeds_counter.keys())
transformers = [paraphrase_maker.get_pattern_chunks, paraphrase_maker.lemmatize_a_word,
                paraphrase_maker.replace_from_general_rules, paraphrase_maker.generate_from_regex]

while min(paraphrase_maker.non_seeds_counter.values()) < int(args.end_number):
    select_labels = [
        label for label in select_labels if paraphrase_maker.non_seeds_counter[label] < int(args.end_number)]
    select_label = select_labels[0]
    transformer = random.choice(transformers)
    paraphrases_for_label = paraphrase_maker.label2paraphrases[select_label]
    paraphrase_in = random.sample(paraphrases_for_label, 1)[0]
    new_paraphrase = paraphrase_maker.create_paraphrase(
        paraphrase_in, transformer)

    if new_paraphrase.text not in paraphrase_maker.existing_texts:
        paraphrase_maker.paraphrase_adder(new_paraphrase)





paraphrase_maker.df.to_csv(sys.stdout)
