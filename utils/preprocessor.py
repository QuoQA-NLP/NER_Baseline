

class Preprocessor :

    def __init__(self, ) :
        self.mapping = {
            0: 0,       # B-PS
            1: 0,       
            2: 1,       # B-LC
            3: 1,       
            4: 2,       # B-OG
            5: 2,
            6: 3,       # B-DT
            7: 3,
            8: 4,       # B-IT
            9: 4,
            10: 5,      # B-QT
            11: 5,
            12: 6,      # O
            -100: -100, # LABEL_PAD_TOKEN
        }

    def __call__(self, datasets) :
        ner_tags = datasets["labels"]
        batch_size = len(ner_tags)

        p_tags = []
        for i in range(batch_size):
            ner_tag = ner_tags[i]

            tag = [self.mapping[t] for t in ner_tag]
            p_tags.append(tag)

        datasets["labels"] = p_tags
        return datasets
