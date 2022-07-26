
class Postprocessor : 
    def __init__(self, ) :
        self.label_mapping = {
            0: "B-PS",
            1: "B-LC",
            2: "B-OG",
            3: "B-DT",
            4: "B-TI",
            5: "B-QT",
            6: "O",
            7: "I-PS",
            8: "I-LC",
            9: "I-OG",
            10: "I-DT",
            11: "I-TI",
            12: "I-QT",
        }

    def recover(self, p) :
        size = len(p)
        ans_list = [p[0]]
        for i in range(1, size) :

            if (p[i-1] == "B-PS" or p[i-1] == "I-PS") and p[i] == "B-PS" :
                p[i] = "I-PS"

            if (p[i-1] == "B-LC" or p[i-1] == "I-LC") and p[i] == "B-LC" :
                p[i] = "I-LC"

            if (p[i-1] == "B-OG" or p[i-1] == "I-OG") and p[i] == "B-OG" :
                p[i] = "I-OG"

            if (p[i-1] == "B-DT" or p[i-1] == "I-DT") and p[i] == "B-DT" :
                p[i] = "I-DT"

            if (p[i-1] == "B-TI" or p[i-1] == "I-TI") and p[i] == "B-TI" :
                p[i] = "I-TI"

            if (p[i-1] == "B-QT" or p[i-1] == "I-QT") and p[i] == "B-QT" :
                p[i] = "I-QT"

            ans_list.append(p[i])
        return ans_list


    def __call__(self, predictions, labels) :
        predictions = [[self.label_mapping[p] for p in pred] for pred in predictions]
        predictions = [self.recover(pred) for pred in predictions]

        labels = [[self.label_mapping[l] for l in label] for label in labels]

        return predictions, labels
