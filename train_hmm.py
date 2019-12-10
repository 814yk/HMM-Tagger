import sys
import re

from collections import defaultdict

# Files
TAG_FILE = sys.argv[1]
TOKEN_FILE = sys.argv[2]
OUTPUT_FILE = sys.argv[3]

# Vocabulary
vocab = {}
OOV_WORD = "OOV"
INIT_STATE = "init"
FINAL_STATE = "final"

# Transition and emission probabilities
emissions = {}
transitions = {}
transitions2={}
transitions_total = defaultdict(lambda: 0)
#transitions_total2 = defaultdict(lambda: 0)
emissions_total = defaultdict(lambda: 0)

with open(TAG_FILE) as tag_file, open(TOKEN_FILE) as token_file:
    for tag_string, token_string in zip(tag_file, token_file):
        tags = re.split("\s+", tag_string.rstrip())
        tokens = re.split("\s+", token_string.rstrip())
        pairs = zip(tags, tokens)
        tokens[0]=tokens[0].lower()
        prevtag = INIT_STATE
        prevtag2= INIT_STATE
        for (tag, token) in pairs:

            # this block is a little trick to help with out-of-vocabulary (OOV)
            # words.  the first time we see *any* word token, we pretend it
            # is an OOV.  this lets our model decide the rate at which new
            # words of each POS-type should be expected (e.g., high for nouns,
            # low for determiners).

            if token not in vocab:
                vocab[token] = 1
                token = OOV_WORD
                
            if prevtag2 not in transitions2:
                transitions2[prevtag2] = defaultdict()
            if prevtag not in transitions2[prevtag2]:
                transitions2[prevtag2][prevtag] = defaultdict(lambda: 0)
            if tag not in emissions:
                emissions[tag] = defaultdict(lambda: 0)
            if prevtag not in transitions:
                transitions[prevtag] = defaultdict(lambda: 0)


            emissions[tag][token] += 1
            emissions_total[tag] += 1
            
            transitions[prevtag][tag] += 1
            transitions2[prevtag2][prevtag][tag] +=1
            #transitions_total2[prevtag2][prevtag]+=1
            transitions_total[prevtag] += 1
            prevtag2=prevtag
            prevtag = tag

        if prevtag not in transitions:
            transitions[prevtag] = defaultdict(lambda: 0)
        if prevtag2 not in transitions2:
            transitions2[prevtag2] = defaultdict()
        if prevtag not in transitions2[prevtag2]:
            transitions2[prevtag2][prevtag] = defaultdict(lambda: 0)


        transitions2[prevtag2][prevtag][FINAL_STATE] += 1
        transitions[prevtag][FINAL_STATE] += 1
        transitions_total[prevtag] += 1
        
# Total
with open(OUTPUT_FILE, "w") as f:
    for prevtag2 in transitions2:
        for prevtag in transitions2[prevtag2]:
            for tag in transitions2[prevtag2][prevtag]:
                tri_score=transitions2[prevtag2][prevtag][tag] / sum(transitions2[prevtag2][prevtag].values())
                f.write("tri-trans {} {} {} {}\n"
                    .format(prevtag2,prevtag, tag, tri_score))
                
    for prevtag in transitions:
        for tag in transitions[prevtag]:
            bi_score=transitions[prevtag][tag] / transitions_total[prevtag]
            uni_score=emissions_total[tag] / sum(emissions_total.values())
            f.write("bi-trans {} {} {}\n"
            .format(prevtag, tag, bi_score))
                
    for tag in emissions_total:
        uni_score=emissions_total[tag] / sum(emissions_total.values())
        f.write("uni-trans {} {}\n"
        .format(tag, uni_score))
            
    for tag in emissions:
        for token in emissions[tag]:
            f.write("emit {} {} {}\n"
                .format(tag, token, emissions[tag][token] / emissions_total[tag]))
