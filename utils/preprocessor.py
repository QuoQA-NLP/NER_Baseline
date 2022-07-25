

class Preprocessor :

    def __init__(self, ) :
        self.mapping = {
            0: 1,       # B-PS
            1: 1,       
            2: 2,       # B-LC
            3: 2,       
            4: 3,       # B-OG
            5: 3,
            6: 4,       # B-DT
            7: 4,
            8: 5,       # B-IT
            9: 5,
            10: 6,      # B-QT
            11: 6,
            12: 0,      # O
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
