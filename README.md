generate_paraphrases.py is meant to help in the creation of high quality, varied, and grammatical paraphrases for multiple intents at once. The paraphrases are generated through iterative applications of rules and transformations. Each intent must have a unique intent label, (i.e. no_points_tx_00) and at least one canonical utterance that represents that intent (i.e. "are there any transactions i can't earn points from"). The number of paraphrases that can be created depends on how much material is added to the required inputs for the script. The sample input files (general_rules_tsv.tsv, generate_paraphrases.py, patterns_json.json, regex_tsv.tsv, seeds_tsv.tsv) can be entered into the script to create a small demo for around 50-100 unique paraphrases.

INPUT FILES:
  --regex_tsv REGEX_TSV
                        regex tsv of regular expressions that correspond to
                        each intent, columns: 1) intent label 2) regular
                        expression that generates a paraphrase for that intent
  --general_rules_tsv GENERAL_RULES_TSV
                        general rules tsv with columns: 1) regular expression
                        2) possible replacement strings separated by '||'
  --seeds_tsv SEEDS_TSV
                        tsv with columns: 1) intent label 2) canonical
                        paraphrase for that intent
  --end_number END_NUMBER
                        number of paraphrases required for each intent
  --patterns_json PATTERNS_JSON
                        json of spacy matching patterns for short sentence and
                        phrase patterns within existing paraphrases

See sample input files for the proper formatting, etc.

RUNNING THE DEMO:
To run the demo, you must enter all the required inputs, like this: 
	python generate_paraphrases.py --regex_tsv regex_tsv.tsv --general_rules_tsv general_rules_tsv.tsv --patterns_json patterns_json.json --seeds_tsv seeds_tsv.tsv --end_number 50 > out.tsv


CREATING A NEW INTENT:
To create a new intent, at least one canonical seed must be added to seeds_tsv.tsv and some rules should be added to regex_tsv.tsv as well. The remaining input files do not necessarily need to be updated, as they apply generally to any sort of question paraphrases.

OUTPUT CSV:
The script outputs a csv to stdout with the canonical utterances and paraphrases together along with their respective labels and some information showing how the paraphrase had been generated. For example: 
	postmates_04,usage of postmates,get_pattern_chunks,what is the usage of postmates
In this case "usage of postmates" was generated as a chunk parse (chunks defined in patterns_json argument)

